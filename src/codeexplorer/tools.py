from pathlib import Path
import subprocess
from typing import Dict, List

#
# Helpers
#


def is_in_git_repo(directory: Path) -> bool:
    try:
        status = subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return status.returncode == 0
    except Exception as e:
        print(f"Error checking git status: {e}")
        return False


def git_ls_files(directory: Path) -> List[str]:
    """
    Returns a list of files under git's control.

    Generally provides a much better listing of "the source code"
    when in a git repo.
    """
    try:
        status = subprocess.run(  # noqa: F821
            ["git", "-C", str(directory), "ls-files", "-c"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if status.returncode == 128:
            # Exit code 128 means "not a git repository" or
            # non-existent directory
            return []
        if status.returncode != 0:
            return []

        return status.stdout.splitlines()

    except Exception as e:
        print(f"Error checking git status: {e}")
        return []


#
# Tools
#


def read_file_path(jail: Path, path: str) -> str:
    p = Path(path).absolute()
    if p != jail and jail not in p.parents:
        return f"ERROR: Path {p} must have {jail} as an ancestor"
    if not p.exists():
        return f"ERROR: Path {path} does not exist"
    if not p.is_file():
        return f"ERROR: Path {path} is not a file"
    with open(path, "r") as f:
        return f.read()


def list_directory_simple(jail: Path, path: str) -> str:
    """
    Return a list of files in the directory and subdirectories as a
    list of absolute file paths, one path per line.
    """
    p = Path(path).absolute()
    if p != jail and jail not in p.parents:
        return f"ERROR: Path {path} must have {jail} as an ancestor"
    if not p.exists():
        return f"ERROR: Path {p} does not exist"
    if not p.is_dir():
        return f"ERROR: Path {p} is not a directory"

    if is_in_git_repo(p):
        fs = git_ls_files(p)
        return "\n".join([str(p / f) for f in fs])
    else:
        result = []

        def _tree(dir_path: Path):
            for path in [x for x in dir_path.iterdir() if x.is_file()]:
                if path.name.startswith("."):
                    continue
                result.append(str(path.absolute()))

            for path in [x for x in dir_path.iterdir() if x.is_dir()]:
                if path.name.startswith("."):
                    continue
                if path.is_dir():
                    _tree(path)

        _tree(p)
        result = sorted(result)
        return "\n".join(result)


def str_replace(jail: Path, path: str, old_str: str, new_str: str) -> str:
    p = Path(path).absolute()
    if p != jail and jail not in p.parents:
        return f"ERROR: Path {path} must have {jail} as an ancestor"
    if not p.exists():
        return f"ERROR: Path {path} does not exist"
    if not p.is_file():
        return f"ERROR: Path {path} is not a file"
    with open(path, "r") as f:
        content = f.read()

    occurances = content.count(old_str)
    if occurances == 0:
        return (
            "Error: No match found for replacement."
            + " Please check your text and try again."
        )
    if occurances > 1:
        return (
            f"Error: Found {occurances} matches for replacement text."
            + " Please provide more context to make a unique match."
        )

    new_content = content.replace(old_str, new_str, 1)

    with open(path, "w") as f:
        f.write(new_content)

    return "Success editing file"


def create(jail: Path, path: str, file_text: str) -> str:
    p = Path(path).absolute()
    if p != jail and jail not in p.parents:
        return f"ERROR: Path {path} must have {jail} as an ancestor"
    if p.exists():
        return (
            f"ERROR: Path {path} already exists; choose another file name."
        )
    with open(path, "w") as f:
        f.write(file_text)
    return "Success creating file"


tools = [
    {
        "name": "list_directory_simple",
        "description": """
            List the complete file tree for path, including subdirectories.

            The passed path MUST be a directory, not a file.

            Use this tool to discover the files in the codebase. The returned value is a list of absolute paths. Use these paths to explore the codebase.

            Here is an example tree, with a python project in a src directory:

            /Users/mike/code/pythonapp/src/mypackage/entrypoint.py
            /Users/mike/code/pythonapp/src/mypackage/main.py
            /Users/mike/code/pythonapp/LICENSE
            /Users/mike/code/pythonapp/README.md
            /Users/mike/code/pythonapp/pyproject.toml
            /Users/mike/code/pythonapp/uv.lock

            Real lists are likely to be much larger!
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to list files",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file_path",
        "description": """
            Read a file.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to a file",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "think",
        "description": "Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed. For example, if you explore the repo and discover the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective. Alternatively, if you receive some test results, call this tool to brainstorm ways to fix the failing tests.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your thoughts.",
                }
            },
            "required": ["thought"],
        },
    },
]

edit_tools = [
    {
        "name": "str_replace",
        "description": """
        Edit a file by specifying an exact existing string and its replacement.
    """,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to a file",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact string to replace",
                },
                "new_str": {
                    "type": "string",
                    "description": "Exact replacement string",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "create",
        "description": """
        Create a file, supplying its contents.

        If the file already exists, this function will fail.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to a file.",
                },
                "file_text": {
                    "type": "string",
                    "description": "Text content for new file",
                },
            },
            "required": ["path"],
        },
    },
]


def process_tool_call(
    jail: Path, tool_name: str, tool_input: Dict[str, str]
) -> str:
    match tool_name:
        case "list_directory_simple":
            return list_directory_simple(jail, tool_input["path"])
        case "read_file_path":
            return read_file_path(jail, tool_input["path"])
        case "str_replace":
            return str_replace(
                jail,
                tool_input["path"],
                tool_input["old_str"],
                tool_input["new_str"],
            )
        case "create":
            return create(
                jail,
                tool_input["path"],
                tool_input["file_text"],
            )
        case "think":
            return tool_input["thought"]
    return f"Error: no tool with name {tool_name}"
