import logging
import os
import re
import subprocess
import time

import chardet
import paramiko
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    Chroma = None

from Config import *


class InteractiveDockerShell:
    HOSTNAME = "c0mpi1er-c0nta1ner"

    def __init__(
        self,
        local_path,
        image_name="autocompiler:gcc13",
        use_proxy=USE_PROXY,
        stuck_timeout=120,
        cmd_timeout=3600,
        pre_exec=False,
    ):
        container_id = None
        self.client = None
        self.session = None
        self.container_id = None
        self.use_proxy = bool(use_proxy)
        self.stuck_timeout = stuck_timeout
        self.cmd_timeout = cmd_timeout
        self.last_line = "root@c0mpi1er-c0nta1ner:/work# "
        self.logger = []
        self._cmd_counter = 0

        abs_local_path = os.path.abspath(local_path)

        try:
            run_cmd = [
                "docker",
                "run",
                "--network",
                "bridge",
                "--hostname",
                self.HOSTNAME,
                "-p",
                "127.0.0.1::22",
                "-v",
                f"{abs_local_path}:/work/",
                "-itd",
                image_name,
                "/bin/bash",
            ]
            run_res = subprocess.run(run_cmd, capture_output=True, text=True)
            container_id = run_res.stdout.strip()
            if run_res.returncode != 0 or len(container_id) != 64:
                raise RuntimeError(
                    f"Failed to create container (rc={run_res.returncode}): "
                    f"{run_res.stderr.strip() or run_res.stdout.strip()}"
                )
            logging.info(f"[+] Container {container_id} created.")

            ssh_start = subprocess.run(
                ["docker", "exec", container_id, "service", "ssh", "start"],
                capture_output=True,
                text=True,
            )
            if ssh_start.returncode != 0:
                raise RuntimeError(
                    f"Could not start SSH: {ssh_start.stdout}\n{ssh_start.stderr}"
                )

            port_res = subprocess.run(
                ["docker", "port", container_id, "22/tcp"],
                capture_output=True,
                text=True,
                check=True,
            )
            port_output = port_res.stdout.strip().splitlines()[0]
            ssh_port = int(port_output.rsplit(":", 1)[1])

            # Wait for sshd to report running
            for _ in range(20):
                ssh_check = subprocess.run(
                    ["docker", "exec", container_id, "service", "ssh", "status"],
                    capture_output=True,
                    text=True,
                )
                if ssh_check.returncode == 0:
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError(
                    f"SSH did not become ready: {ssh_check.stdout}\n{ssh_check.stderr}"
                )

            # Connect via Paramiko with brief retry in case port forwarding is warming up
            last_ssh_err = None
            for _ in range(5):
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                try:
                    client.connect(
                        "127.0.0.1",
                        username="root",
                        password=PASSWORD or "root",
                        port=ssh_port,
                        timeout=15,
                    )
                    self.client = client
                    break
                except Exception as ssh_err:
                    last_ssh_err = ssh_err
                    try:
                        client.close()
                    except Exception:
                        pass
                    time.sleep(0.5)
            else:
                raise RuntimeError(f"Paramiko SSH connect failed: {last_ssh_err}")

            self.session = self.client.invoke_shell(width=200, height=50)
            self.session.settimeout(stuck_timeout)
            time.sleep(0.5)
            while self.session.recv_ready():
                self.session.recv(4096)

            self.container_id = container_id
        except Exception as e:
            if self.client:
                try:
                    self.client.close()
                except Exception:
                    pass
                self.client = None
            if container_id and len(container_id) >= 12:
                logging.warning(f"[-] Cleaning up failed container {container_id[:12]}...")
                subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
            raise Exception(f"Failed to create the container, error: {str(e)}")

        if pre_exec and self.use_proxy:
            self.execute_command("proxychains -q apt-get update")
        elif pre_exec:
            self.execute_command("apt-get update")

    @staticmethod
    def _clean_command(command: str) -> str:
        command = command.strip()
        if command.startswith("```") and command.endswith("```"):
            inner = command[3:-3].strip()
            lines = inner.splitlines()
            if lines and lines[0].strip().lower() in ("bash", "sh", "shell", "zsh"):
                inner = "\n".join(lines[1:]).strip()
            command = inner
        for wrap in ("**", "`"):
            if command.startswith(wrap) and command.endswith(wrap) and len(command) > len(wrap) * 2:
                command = command[len(wrap) : -len(wrap)].strip()
        for quote in ('"', "'"):
            if (
                command.startswith(quote)
                and command.endswith(quote)
                and len(command) > 2
                and quote not in command[1:-1]
            ):
                command = command[1:-1].strip()
        in_quote = None
        i = 0
        while i < len(command):
            ch = command[i]
            if in_quote is None:
                if ch == "\\":
                    i += 2
                    continue
                if ch in ('"', "'"):
                    in_quote = ch
            elif in_quote == "'":
                if ch == "'":
                    in_quote = None
            elif in_quote == '"':
                if ch == "\\" and i + 1 < len(command) and command[i + 1] in ('"', "\\"):
                    i += 2
                    continue
                if ch == '"':
                    in_quote = None
            i += 1
        if in_quote is not None:
            command += in_quote
        return command

    def _drain_ready(self) -> str:
        chunks = []
        while self.session and self.session.recv_ready():
            recv = self.session.recv(4096)
            if not recv:
                break
            chunks.append(recv.decode(encoding="utf-8", errors="ignore"))
        return "".join(chunks)

    @staticmethod
    def _is_make_install_command(command: str) -> bool:
        for seg in re.split(r"\s*(?:&&|\|\||;|\||\n)\s*", command):
            stripped = seg.strip()
            if re.match(
                r"^(?:[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|\S+)\s+)*make\b",
                stripped,
            ) and re.search(r"\sinstall(?:\s|$)", stripped):
                return True
        return False

    @staticmethod
    def _normalize_apt_command(command: str) -> str:
        parts = re.split(r"(\s*(?:&&|\|\||;|\||\n)\s*)", command)
        for idx in range(0, len(parts), 2):
            seg = parts[idx]
            if not re.search(r"\bapt(?:-get)?\s+install\b", seg):
                continue
            if not re.search(r"(?:\s|^)(?:-y|--yes)(?:\s|$)", seg):
                seg = re.sub(r"\b(apt(?:-get)?\s+install)\b", r"\1 -y", seg, count=1)
            if "DEBIAN_FRONTEND=" not in seg:
                leading = len(seg) - len(seg.lstrip())
                seg = f"{seg[:leading]}DEBIAN_FRONTEND=noninteractive {seg[leading:]}"
            parts[idx] = seg
        return "".join(parts)

    @staticmethod
    def _normalize_make_jobs(command: str, max_jobs: int = MAKE_JOBS) -> str:
        skip_targets = {"clean", "distclean", "mrproper", "install", "uninstall", "check", "test", "help"}
        parts = re.split(r"(\s*(?:&&|\|\||;|\||\n)\s*)", command)
        make_prefix_re = re.compile(
            r"^(\s*(?:\(\s*)?(?:[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|\S+)\s+)*(?:proxychains(?:\s+-q)?\s+)?)(make\b.*)$",
            re.DOTALL,
        )
        for idx in range(0, len(parts), 2):
            seg = parts[idx]
            m = make_prefix_re.match(seg)
            if not m:
                if re.search(r"\bcmake\s+--build\b", seg):
                    seg = re.sub(r"-j\s*\$\([^)]+\)", f"-j{max_jobs}", seg)
                    parts[idx] = seg
                continue

            prefix, make_part = m.group(1), m.group(2)
            make_part = re.sub(r"-j\s*\$\([^)]+\)", f"-j{max_jobs}", make_part)
            make_part = re.sub(
                r"-j\s*(\d+)\b",
                lambda jm: f"-j{max_jobs}" if int(jm.group(1)) > max_jobs else jm.group(0),
                make_part,
            )
            make_part = re.sub(r"\s-j(?=[\s)]|$)", f" -j{max_jobs}", make_part)
            if not re.search(r"(?:^|\s)-j(?:\s*\d+|\s|$)", make_part):
                tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_-]*\b", make_part))
                if not (tokens & skip_targets):
                    make_part = re.sub(r"^make\b", f"make -j{max_jobs}", make_part, count=1)
            parts[idx] = f"{prefix}{make_part}"
        return "".join(parts)

    def execute_command(self, command: str, record_log: bool = True) -> str:
        """
        Execute a command in an interactive shell on the local container (Debian GNU/Linux 12 with GCC 13.2.0 and root user). Initially, we are in the /work/ directory.
        Note: APT package lists are not pre-cached; always run 'apt-get update && apt-get install -y <pkg>' when installing missing libraries or headers.
        @param command: The command to execute.
        """
        if self.session is None:
            raise Exception("No session available.")

        command = self._clean_command(command)
        if not command:
            return ""

        if command.strip() == "^C":
            self.session.send("\x03")
            time.sleep(0.5)
            output = self._drain_ready()
            return self.omit("^C", output, 0.5, record_log=record_log)

        if self._is_make_install_command(command):
            return "Tips: Do not install the project, just compile it!"

        if self.use_proxy:
            if "git " in command and "proxychains" not in command:
                command = command.replace("git ", "proxychains -q git ")
            if "curl " in command and "proxychains" not in command:
                command = command.replace("curl ", "proxychains -q curl ")
            if "wget " in command and "proxychains" not in command:
                command = command.replace("wget ", "proxychains -q wget ")

        command = self._normalize_apt_command(command)
        command = self._normalize_make_jobs(command, max_jobs=MAKE_JOBS)

        # Flush any leftover prompt bytes before sending next command
        self._drain_ready()

        command = command.rstrip("\\ \t\r\n")
        self._cmd_counter += 1
        sentinel = f"__AC_DONE_{os.getpid()}_{self._cmd_counter}_{int(time.time() * 1000)}__"
        sentinel_re = re.compile(rf"{re.escape(sentinel)}:(\d+):__END__")
        payload = f"{{ {command}\n}}; __ac_ec=$?; printf '\\n%s:%d:__END__\\n' \"{sentinel}\" $__ac_ec\n"
        self.session.send(payload)

        cmd_start_time = time.time()
        last_activity = cmd_start_time
        output = ""
        exit_code = None

        while True:
            if self.session.recv_ready():
                output += self._drain_ready()
                last_activity = time.time()

            match = sentinel_re.search(output)
            if match:
                exit_code = int(match.group(1))
                # Wait briefly for trailing prompt bytes after sentinel line
                time.sleep(0.1)
                if self.session.recv_ready():
                    output += self._drain_ready()
                return self.omit(
                    command,
                    output,
                    time.time() - cmd_start_time,
                    sentinel=sentinel,
                    exit_code=exit_code,
                    record_log=record_log,
                )

            now = time.time()
            if now - cmd_start_time > self.cmd_timeout or now - last_activity > self.stuck_timeout:
                self.session.send("\x03")
                flag = "\nCommand execution timeout!\n"
                stop_wait_start = time.time()
                while time.time() - stop_wait_start < 5.0:
                    if self.session.recv_ready():
                        output += self._drain_ready()
                        match = sentinel_re.search(output)
                        if match:
                            exit_code = int(match.group(1))
                            break
                        if self.HOSTNAME in output:
                            break
                    else:
                        time.sleep(0.1)
                else:
                    flag = (
                        "\nShell timed out or is waiting for interactive input. "
                        "You can send input or ^C via the Shell tool.\n"
                    )
                output += flag
                return self.omit(
                    command,
                    output,
                    time.time() - cmd_start_time,
                    sentinel=sentinel,
                    exit_code=exit_code,
                    record_log=record_log,
                )

            time.sleep(0.05)

    def omit(
        self,
        command: str,
        output: str,
        duration: float,
        sentinel: str = None,
        exit_code: int = None,
        record_log: bool = True,
    ) -> str:
        """
        Clean ANSI escapes and sentinel lines, update prompt state, and format output.
        """
        # Strip ANSI escape sequences and normalize CRLF
        output = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", output)
        output = output.replace("\r\n", "\n").replace("\r", "\n")

        if sentinel:
            # Remove the echoed printf helper command and the sentinel marker line
            cleaned_lines = []
            for line in output.split("\n"):
                if sentinel in line:
                    continue
                cleaned_lines.append(line)
            output = "\n".join(cleaned_lines)

        output = self.last_line + output
        if record_log:
            self.logger.append([command, output, duration])

        lines = output.split("\n")
        self.last_line = lines[-1] if lines else ""

        if command.startswith("make"):
            result = "\n".join(lines[-50:])
        elif command.startswith("configure") or command.startswith("./configure"):
            result = "\n".join(lines[-30:])
        elif command.startswith("cmake"):
            result = "\n".join(lines[-30:])
        else:
            if len(output) > 8000:
                result = output[:4000] + "\n......\n" + output[-4000:]
            else:
                result = output

        if exit_code is not None:
            result = f"{result.rstrip()}\n[Exit code: {exit_code}]"
        return result

    def close(self):
        logging.info("[-] Stopping docker container, please wait a second...")
        if self.session:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
        if self.container_id:
            subprocess.run(["docker", "rm", "-f", self.container_id], capture_output=True)
            self.container_id = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# RAG and Model decision
class SearchCompilationInstruction:
    def __init__(self, directory_path: str, project_name: str, threshold=0.80, use_proxy=False):
        self.vectorstore = None
        self.similarity_threshold = threshold
        self.directory_path = directory_path
        self.project_name = project_name
        self.vec_store = f"vec_store/{self.project_name}"
        self.compilation_ins_doc = []
        self.logger = []
        self.logger1 = []
        self.logger2 = []
        self.logger3 = []
        self.use_proxy = use_proxy
        self.startswith_list = ["readme", "build", "install", "contributing", "how-to"]
        self.endswith_list = [".markdown"]
        self.file_name = []
        self.keywords = ["compile", "build", "compilation"]
        self.keywords_files = []
        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                if any(file.lower().startswith(f"{prefix}") for prefix in self.startswith_list):
                    self.compilation_ins_doc.append(os.path.join(root, file))
                if any(file.lower().endswith(f"{suffix}") for suffix in self.endswith_list):
                    self.compilation_ins_doc.append(os.path.join(root, file))
                if file.lower() in self.file_name:
                    self.compilation_ins_doc.append(os.path.join(root, file))

    def read_files(self, doc_path_list: list) -> list:
        documents = []
        for file_path in doc_path_list:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                    documents.append(
                        Document(page_content=fp.read(), metadata={"source": file_path})
                    )
            except Exception as e:
                logging.error(f"[!] Failed to read {file_path}:\n{e}")
        return documents

    def setup_rag(self, docs):
        if Chroma is None:
            logging.warning("[!] langchain_chroma is not installed; skipping RAG vectorstore setup.")
            return False, 0
        if docs != []:
            try:
                start_time = time.time()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800, chunk_overlap=80, add_start_index=True
                )
                all_splits = text_splitter.split_documents(docs)
                emb_function = OpenAIEmbeddings(
                    base_url=OPENAI_BASE_URL,
                    model=OPENAI_EMBEDDING_MODEL,
                    api_key=OPENAI_API_KEY,
                    check_embedding_ctx_length=False,
                )
                if os.path.exists(self.vec_store):
                    self.vectorstore = Chroma(
                        persist_directory=self.vec_store, embedding_function=emb_function
                    )
                    return True, 0
                else:
                    self.vectorstore = Chroma.from_documents(
                        documents=all_splits,
                        embedding=emb_function,
                        persist_directory=self.vec_store,
                    )
                end_time = time.time()
                return True, (end_time - start_time)
            except Exception as e:
                logging.error(f"[!] Failed to build vectorstore:\n{e}")
                return False, 0
        return False, 0

    def get_relevant_docs(self, query):
        docs_and_scores = self.vectorstore.similarity_search_with_score(query)
        relevant_docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc, score in docs_and_scores
            if score >= self.similarity_threshold
        ]
        return relevant_docs

    def search_instruction(self, rag_ok):
        self.logger = []
        if rag_ok:
            query = "How to compile/build the project?"
            docs = self.get_relevant_docs(query)
            if docs:
                try:
                    llm = create_chat_llm()
                    template = """You are an experienced software development engineer and specialized in extracting compilation commands. The documents from a project repository will be provided and you need to carefully identify relevant compilation/building instructions. If no compilation commands are found, respond with "No compilation commands found." If found, list the compilation commands concisely, clearly, and professionally without any additional explanations.
                    Documents: {text}
                    Answer: """
                    context = "\n".join(doc.page_content for doc in docs)
                    prompt = PromptTemplate.from_template(template=template)
                    rag_chain = (
                        {"text": RunnableLambda(lambda x: context)}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    answer = rag_chain.invoke({})
                    self.logger.append(
                        [
                            template.format(text=context),
                            answer,
                            [[doc.metadata, doc.page_content] for doc in docs],
                            {"ori_content_len": len(context), "answer_len": len(answer)},
                        ]
                    )
                    return answer
                except Exception as e:
                    logging.error(f"[!] Failed search instruction:\n{e}")
                    return "Search failed due to unknown reason."

        return "Not found possible compilation guidance from files stored in the local path."

    def search_url_from_files(self, *args, **kwargs):
        query = "From which URL can I find the compilation instructions?"
        docs = self.get_relevant_docs(query)
        if docs:
            try:
                llm = create_chat_llm()
                template = """You are an experienced software development engineer and specialized in identifying URLs related to compilation commands. Analyze the given text and extract any URLs specifically associated with obtaining or referencing compilation instructions. If no relevant URLs are found, simply state 'No relevant URLs found.' Ensure the result is accurate, concise, and professional.
                Text: {text}
                Answer: """
                context = "\n".join(doc.page_content for doc in docs)
                prompt = PromptTemplate.from_template(template=template)
                rag_chain = (
                    {"text": RunnableLambda(lambda x: context)}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                answer = rag_chain.invoke({})
                self.logger2.append(
                    [
                        template.format(text=context),
                        answer,
                        [[doc.metadata, doc.page_content] for doc in docs],
                        {"ori_content_len": len(context), "answer_len": len(answer)},
                    ]
                )
                return answer
            except Exception as e:
                logging.error(f"[!] Failed search url:\n{e}")
                return "Search failed due to unknown reason."

        return "Not found any relevant URL from files stored in the local path."

    def get_url_content(self, url):
        headers = build_http_headers()
        proxies = resolve_http_proxies(self.use_proxy)

        try:
            response = requests.get(url, headers=headers, proxies=proxies, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            texts = soup.get_text(separator="\n", strip=True)
            return [Document(page_content=texts, metadata={"source": url})]
        except Exception as e:
            logging.error(e)
            return f"Request url error: {url}"

    def search_instruction_by_agent(self, file_path: str) -> str:
        file_path = file_path.strip()
        if ", " in file_path:
            return "The input should be a single file path"
        if not (file_path.startswith("/") or os.path.isabs(file_path)):
            return "The file path should be absolute path."
        if file_path.startswith("/work/"):
            rel_path = file_path[len("/work/") :].lstrip("/")
            real_file_path = os.path.normpath(os.path.join(self.directory_path, rel_path))
        elif file_path == "/work":
            real_file_path = os.path.normpath(self.directory_path)
        elif file_path.startswith("/"):
            real_file_path = os.path.normpath(
                os.path.join(self.directory_path, file_path.lstrip("/"))
            )
        else:
            real_file_path = os.path.normpath(file_path)

        if not os.path.exists(real_file_path):
            return f"File {file_path} does not exist."

        try:
            with open(real_file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if len(content) > 32000:
                    content = content[:32000]
            llm = create_chat_llm()
            template = """You are an experienced software development engineer and specialized in building a project from source code. The content of a file from a project repository will be provided, and you need to carefully analyze and output the useful information about "how to compile the project on linux". The output should be an extraction of the compilation guide section of the file, complete with compilation-related information. If there is no such information, output "<NOT-FOUND-INSTRUCTION>". Make sure the output is complete, accurate, and without additional explanations.
            Document: {file_path}
            {text}
            Output: """
            prompt = PromptTemplate.from_template(template=template)
            chain = prompt | llm | StrOutputParser()
            logging.info(f"[+] Invoke SearchCompilationInstruction for file: {real_file_path}")
            answer = chain.invoke({"text": content, "file_path": file_path})
            self.logger.append(
                [
                    real_file_path,
                    template.format(text=content, file_path=file_path),
                    answer,
                    {"ori_content_len": len(content), "answer_len": len(answer)},
                ]
            )
            return answer
        except Exception as e:
            logging.error(f"[!] Failed search file: {real_file_path}\n{str(e)}")
            return "Search failed due to unknown reason."

    def search_instruction_from_files(self, *args, **kwargs):
        if self.compilation_ins_doc != []:
            files_content = self.read_files(self.compilation_ins_doc)
            rag_ok, duration = self.setup_rag(docs=files_content)
            result = self.search_instruction(rag_ok)
            self.logger1 = self.logger
            self.logger1.append(duration)
            return result
        return "No compilation commands found."

    def search_instruction_from_url(self, url: str, *args, **kwargs):
        if url:
            texts = self.get_url_content(url=url)
            rag_ok, _ = self.setup_rag(docs=texts if isinstance(texts, list) else [])
            result = self.search_instruction(rag_ok)
            self.logger3 = self.logger
            return result
        return "No compilation commands found."
