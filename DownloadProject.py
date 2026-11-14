import logging
import os
import shutil
import subprocess
import uuid


def download_project(url, local_path, download_proxy=None, timeout=180) -> bool:
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
        local_path,
    ]
    if download_proxy:
        cmd.extend(["--config", f"http.proxy={download_proxy}"])
    logging.info(f"[-] Running command: {' '.join(cmd)}")
    ret = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if ret.returncode != 0:
        err_msg = ret.stderr.decode(encoding="utf-8", errors="replace")
        raise Exception(f"Failed to download project from {url}\n\n{err_msg}")
    return True


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


def copy_project(local_path):
    project_uuid = uuid.uuid4()
    local_path = os.path.abspath(local_path)
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
        raise Exception(f"Failed to copy project from {local_path} to {new_path}: {e}")
    return new_path
