import logging
import os
import shutil
import stat
import subprocess
import sys
import uuid


def _safe_rmtree(path: str) -> None:
    if not path or not os.path.exists(path):
        return

    def _on_error(func, func_path, _exc_info):
        try:
            os.chmod(func_path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass
        try:
            func(func_path)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_error)
    else:
        shutil.rmtree(path, onerror=_on_error)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def download_project(url, local_path, download_proxy=None, timeout=180) -> bool:
    local_path = os.path.abspath(local_path)
    parent_dir = os.path.dirname(local_path)
    os.makedirs(parent_dir, exist_ok=True)
    tmp_path = f"{local_path}.tmp-{uuid.uuid4().hex[:8]}"

    logging.info(f"[-] Downloading project from {url} to {local_path}")
    cmd = [
        "git",
        "clone",
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.eol=lf",
        "--depth",
        "1",
        url,
        tmp_path,
    ]
    if download_proxy:
        cmd.extend(["--config", f"http.proxy={download_proxy}"])
    logging.info(f"[-] Running command: {' '.join(cmd)}")

    git_env = os.environ.copy()
    git_env["GIT_TERMINAL_PROMPT"] = "0"
    git_env["GCM_INTERACTIVE"] = "never"
    git_env["GIT_ASKPASS"] = "echo"

    try:
        ret = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=git_env,
        )
        if ret.returncode != 0:
            err_msg = ret.stderr.decode(encoding="utf-8", errors="replace")
            raise RuntimeError(f"Failed to download project from {url}\n\n{err_msg}")
        if os.path.exists(local_path):
            if os.path.isdir(local_path) and not os.listdir(local_path):
                os.rmdir(local_path)
            else:
                raise RuntimeError(f"Target directory already exists and is non-empty: {local_path}")
        shutil.move(tmp_path, local_path)
        return True
    except Exception as exc:
        _safe_rmtree(tmp_path)
        if os.path.isdir(local_path):
            try:
                if not os.listdir(local_path):
                    os.rmdir(local_path)
            except OSError:
                pass
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Failed to download project from {url}: {exc}") from exc


_UNIX_SCRIPT_EXTENSIONS = (".sh", ".pl", ".pm", ".py", ".ac", ".am", ".m4", ".in")
_UNIX_SCRIPT_NAMES = {
    "configure",
    "autogen.sh",
    "buildconf",
    "install-sh",
    "missing",
    "depcomp",
    "compile",
    "config.guess",
    "config.sub",
    "ltmain.sh",
    "makefile",
    "gnumakefile",
}


def _normalize_unix_scripts_lf(root_dir: str) -> int:
    """Convert CRLF to LF on Unix build/script files in the copied workspace only."""
    converted = 0
    for cur_root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d != ".git"]
        for fname in files:
            lower = fname.lower()
            if not (lower in _UNIX_SCRIPT_NAMES or lower.endswith(_UNIX_SCRIPT_EXTENSIONS)):
                continue
            fpath = os.path.join(cur_root, fname)
            if os.path.islink(fpath):
                continue
            try:
                with open(fpath, "rb") as fp:
                    data = fp.read()
                if b"\x00" in data or b"\r\n" not in data:
                    continue
                with open(fpath, "wb") as fp:
                    fp.write(data.replace(b"\r\n", b"\n"))
                converted += 1
            except OSError:
                continue
    return converted


def copy_project(local_path, dest_parent_dir=None):
    project_uuid = uuid.uuid4()
    local_path = os.path.abspath(local_path)
    if dest_parent_dir:
        parent = os.path.abspath(dest_parent_dir)
        os.makedirs(parent, exist_ok=True)
        new_path = os.path.join(parent, f"{os.path.basename(local_path)}-{project_uuid}")
    else:
        new_path = f"{local_path}-{project_uuid}"
    logging.info(f"[-] Copy project from {local_path} to {new_path}")
    try:
        shutil.copytree(
            local_path,
            new_path,
            symlinks=True,
            ignore_dangling_symlinks=True,
        )
        converted = _normalize_unix_scripts_lf(new_path)
        if converted:
            logging.info(f"[-] Normalized CRLF->LF on {converted} build script(s) in {new_path}")
    except Exception as e:
        _safe_rmtree(new_path)
        raise Exception(f"Failed to copy project from {local_path} to {new_path}: {e}")
    return new_path
