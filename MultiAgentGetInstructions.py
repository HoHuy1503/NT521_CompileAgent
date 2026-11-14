import logging
import os
import requests
from bs4 import BeautifulSoup
from langchain.agents import Tool, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from Config import *
from CustomAgentExecutor import CompilationAgentExecutor, SafeReActSingleInputOutputParser


def _make_llm():
    return create_chat_llm()


class AgentTools:
    def __init__(self, local_path: str, project_name, project_structure, proxy):
        self.use_proxy = proxy
        self.project_name = project_name
        self.project_structure = project_structure
        self.project_local_path = os.path.abspath(local_path)
        self.content = ""
        self.logger = []

    def _invoke_chain(self, chain, payload):
        return call_with_rate_limit_retry(chain.invoke, payload)

    def debate(self, file):
        """
        A tool for checking the file which may exist compilation instructions.
        @param file: the file to be checked.
        """
        try:
            llm = _make_llm()
            template = """You are an experienced project developer, and you are good at searching project compilation instructions. Project structure and files which may exist compilation instructions are provided. Please analyze the structure and output the files which may exist compilation instructions.

Project structure: {project_structure}

File: {file}

Output: <files which have been revised>"""
            prompt = PromptTemplate.from_template(template=template)
            chain = prompt | llm | StrOutputParser()
            answer = self._invoke_chain(
                chain, {"project_structure": self.project_structure, "file": file}
            )
            self.logger.append(["debate", file, answer])
            return answer
        except Exception as e:
            logging.warning(f"[!] CompileNavigator.debate failed for '{file}': {e}")
            return str(file)

    def search_instructions_from_files(self, file_path):
        """
        A tool for finding out the compilation instruction from a document file stored in a project repository.
        @param file_path: The absolute path of the document file, e.g. /work/README.md.
        """
        try:
            if not isinstance(file_path, str):
                return "Error: file_path must be a string"
            file_path = file_path.strip()

            for wrapper in ["**", "`", '"', "'"]:
                if file_path.startswith(wrapper) and file_path.endswith(wrapper):
                    file_path = file_path[len(wrapper) : -len(wrapper)]
                if file_path.startswith(wrapper):
                    file_path = file_path[len(wrapper) :]
                if file_path.endswith(wrapper):
                    file_path = file_path[: -len(wrapper)]
            file_path = file_path.strip()

            if ", " in file_path:
                return "The input should be a single file path"
            if not (file_path.startswith("/") or os.path.isabs(file_path)):
                return "The file path should be an absolute path, e.g. /work/README.md."

            # Resolve container-style /work/... paths reliably on both Windows and Linux
            if file_path.startswith("/work/"):
                rel_path = file_path[len("/work/") :].lstrip("/")
                real_file_path = os.path.normpath(
                    os.path.join(self.project_local_path, rel_path)
                )
            elif file_path == "/work":
                real_file_path = os.path.normpath(self.project_local_path)
            elif file_path.startswith("/"):
                real_file_path = os.path.normpath(
                    os.path.join(self.project_local_path, file_path.lstrip("/"))
                )
            else:
                real_file_path = os.path.normpath(file_path)

            if not os.path.exists(real_file_path) or os.path.isdir(real_file_path):
                return f"File {real_file_path} does not exist."

            with open(real_file_path, "r", encoding="utf-8", errors="ignore") as fp:
                content = fp.read()

            if len(content) > 32000:
                content = content[:32000] + "\n... (content truncated due to length)"

            if not content.strip():
                return "File is empty"

            self.content = content

            llm = _make_llm()
            template = """You are an experienced software development engineer and specialized in building a project from source code. The content of a file from a project repository will be provided, and you need to carefully analyze and output the useful information about "how to compile the project on linux from git cloned source code?". You need to follow these rules:

1. The output should be an extraction of the compilation guide section of the file, complete with compilation-related information.
2. If there are any other reference document could have the compilation instruction, you should also mention the file path or the URL of the document in your output.
3. If there is not any such information in it, output "<NOT-FOUND-INSTRUCTION>".
4. Make sure the output is complete, accurate, and without additional explanations.

Document: {file_path}

```
{text}
```

Output: """
            prompt = PromptTemplate.from_template(template=template)
            chain = prompt | llm | StrOutputParser()
            answer = self._invoke_chain(chain, {"text": content, "file_path": file_path})
            self.logger.append(["search_instructions_from_files", file_path, answer])
            return answer
        except Exception as e:
            logging.warning(f"[!] search_instructions_from_files failed for '{file_path}': {e}")
            return f"Error reading file {file_path}: {str(e)}"

    def get_url_from_files(self, *args):
        """
        Retrieve the URL associated with the compilation/building instructions.
        @param: this function takes no parameter.
        """
        if self.content:
            try:
                llm = _make_llm()
                template = """You are an experienced software development engineer and specialized in identifying URLs related to compilation commands. Analyze the given text and extract any URLs specifically associated with obtaining or referencing compilation instructions. If no relevant URLs are found, simply state 'No relevant URLs found.' Ensure the result is accurate, concise, and professional.
                Text: {text}
                Answer:"""
                prompt = PromptTemplate.from_template(template=template)
                chain = prompt | llm | StrOutputParser()
                return self._invoke_chain(chain, {"text": self.content})
            except Exception as e:
                logging.warning(f"[!] get_url_from_files failed: {e}")
                return "Search failed due to unknown reason."
        return "Not found any relevant URL from files stored in the local path."

    def get_content_from_url(self, url):
        """
        Get the web page content from the corresponding url.
        @param url: the URL of the content you want to get
        """
        headers = build_http_headers()
        proxies = resolve_http_proxies(self.use_proxy)

        try:
            response = requests.get(
                url.strip(), headers=headers, proxies=proxies, timeout=20
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            texts = soup.get_text(separator="\n", strip=True)
            if len(texts) > 32000:
                texts = texts[:32000] + "\n... (truncated)"
            self.logger.append(["get_content_from_url", url, texts])
            return texts
        except Exception as e:
            logging.warning(f"[!] get_content_from_url failed for '{url}': {e}")
            return f"Request url error: {url}"


class CompileNavigator:
    def __init__(self, local_path: str, project_name):
        self.local_path = local_path
        self.project_name = project_name
        self.logger = []

    def get_instructions(self, project_structure):
        """
        A tool for finding out the compilation instruction from project structure.
        @param project_structure: The project structure.
        """
        llm = _make_llm()
        agenttools = AgentTools(
            local_path=self.local_path,
            project_name=self.project_name,
            project_structure=project_structure,
            proxy=PROXY,
        )
        tools = [
            Tool(
                name="DEBATE",
                func=agenttools.debate,
                description=agenttools.debate.__doc__,
            ),
            Tool(
                name="SEARCH_INSTRUCTIONS_FROM_FILES",
                func=agenttools.search_instructions_from_files,
                description=agenttools.search_instructions_from_files.__doc__,
            ),
            Tool(
                name="GET_CONTENT_FROM_URL",
                func=agenttools.get_content_from_url,
                description=agenttools.get_content_from_url.__doc__,
            ),
        ]
        prompt = PromptTemplate.from_template(template=template)
        agent = create_react_agent(
            llm=llm,
            prompt=prompt,
            tools=tools,
            output_parser=SafeReActSingleInputOutputParser(),
        )
        agent_executor = CompilationAgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=30,
            verbose=False,
            handle_parsing_errors=True,
        )
        get_instructions_prompt = get_instructions_template.format(
            project_name=self.project_name, project_structure=project_structure
        )

        try:
            answer = call_with_rate_limit_retry(
                agent_executor.invoke, {"input": get_instructions_prompt}
            )
            output_text = answer.get("output", "") if isinstance(answer, dict) else str(answer)
        except Exception as e:
            logging.exception(f"[!] CompileNavigator.get_instructions failed: {e}")
            output_text = f"CompileNavigator encountered an error: {e}"

        self.logger.append([project_structure, output_text, agenttools.logger])
        return output_text
