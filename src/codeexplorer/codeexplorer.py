"""
codeexplorer.py

Implements an agent-model to allow LLM to explore a codebase
using tools, rather than trying to pre-create a large context
from the codebase ourselves.
"""

import argparse
import json
import logging
import signal
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from codeexplorer.adapters import (
    AnthropicAdapter,
    OllamaAdapter,
    WatsonxAdapter,
)


def signal_handler(sig, frame):
    print("Interrupted; exiting.")
    exit(0)


signal.signal(signal.SIGINT, signal_handler)


logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


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
        status = subprocess.run(
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


def is_git_working_copy_clean(directory: Path) -> bool:
    """
    Returns True if working copy is clean or not a git repo, False otherwise
    """
    try:
        status = subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if status.returncode == 128:
            # Exit code 128 means "not a git repository" or
            # non-existent directory
            return True

        # Empty output means working copy is clean
        return status.returncode == 0 and not status.stdout.strip()

    except Exception as e:
        print(f"Error checking git status: {e}")
        return False


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


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Code explorer tool")
    parser.add_argument(
        "-n",
        "--num-turns",
        type=int,
        default=20,
        help="Number of turns (default: 20)",
    )
    parser.add_argument(
        "-c",
        "--chat",
        action="store_true",
        help="Use chat mode to continue chat with model (default: false)",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=["ollama", "anthropic", "watsonx"],
        required=True,
        help="Model provider",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model (default: provider specific)",
    )
    parser.add_argument(
        "-e",
        "--allow-edits",
        action="store_true",
        help="Allow model to create and edit files (default: false)",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default=None,
        help="Task to complete using codebase (default: prompt user for question)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write final output to file (default: write only to terminal)",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to explore (default: current directory)",
    )
    args = parser.parse_args()

    max_turns = args.num_turns
    chat_model = args.model

    # Let model expore this folder for now
    jail = Path(args.path).resolve()

    logger.info(
        "Config: turns: %d, path: %s, model: %s", max_turns, jail, chat_model
    )

    active_tools = tools
    if args.allow_edits:
        if not is_git_working_copy_clean(Path(args.path)):
            print("To edit, git working directory must be clean:", args.path)
            exit(1)
        active_tools.extend(edit_tools)

    openai_tools = [
        {"type": "function", "function": x} for x in active_tools
    ]

    console = Console()

    if not args.task:
        extras = Prompt.ask(
            "Anything you'd like AI to think about (eg, plan how to do X)",
            console=console,
        )
        if extras:
            user_task = extras
        else:
            user_task = "Please explain this codebase"
    else:
        user_task = args.task

    # Prompt notes
    # Hard-coding paragraph comes from Sonnet 4 system card.
    # (via https://simonwillison.net/2025/May/25/claude-4-system-card/)
    prompt = textwrap.dedent(f"""
    You are a programmer's assistant exploring a codebase and carrying out programming tasks.

    Please write a high quality, general purpose solution. If the task is unreasonable or infeasible, or if any of the tests are incorrect, please tell me. Do not hard code any test cases. Please tell me if the problem is unreasonable instead of hard coding test cases!

    You are given access to a git repository and tools to explore list and read files. Use these tools when carrying out the user's task.

    NEVER guess what is in a file! If you can't read the file, tell the user that the tool isn't working.

    The best way to start is to list the files in the project using the list_directory_simple tool. It returns all the files in the directory tree, including subdirectories.

    Once you have the directory listing, check the user provided task and pick some files to look at that seem relevant.

    If the user asks about specific files, make sure to read those files. Take your time and evaluate the code line by line before considering the user provided task.

    If the user asks for updates or edits, make sure you have access to the str_replace and create tools. If you don't, stop working and tell the user right away. Ask them to use `--allow-edits` to provide you the tools.

    If the read_file_path tool fails, double check the path you passed in!

    Take your time and be sure you've looked at everything you need to understand the program and answer the user's task below.

    The project root directory is: {Path(args.path).resolve()}

    Here's the user's task:

    {user_task}
    """)

    console.print(Panel(Markdown(prompt), title="Prompt"))

    # dedented markdown to use when formatting each tool use message

    tool_use_markdown = textwrap.dedent("""
    {text}

    Tool Used: `{tool_name}`

    Tool Input:
    ```json
    {tool_input}
    ```

    Tool Result:
    ```
    {tool_result}
    ```
    """)

    client: OllamaAdapter | AnthropicAdapter | WatsonxAdapter
    try:
        if args.provider == "ollama":
            client = OllamaAdapter(chat_model)
        elif args.provider == "anthropic":
            client = AnthropicAdapter(chat_model)
        elif args.provider == "watsonx":
            client = WatsonxAdapter(chat_model)
        else:
            raise ValueError("Invalid model provider")
    except Exception:
        logger.error("Cannot load provider")
        exit(1)

    chat_history = []
    num_turns = 0

    f = None
    if args.output:
        f = open(Path(args.output).absolute(), "w", encoding="utf-8")

    if args.output and f is None:
        logger.error("Could not open output file; exiting.")
        exit(1)

    while True:
        # The "agent loop" --- loop until the model stops
        # requesting tools
        for _ in range(0, max_turns):
            num_turns += 1
            logger.debug(f"\n{'=' * 50}")

            messages = client.prepare_messages(prompt, chat_history)

            with console.status("Model is working..."):
                logger.debug("Messaging model")
                chat_response = client.chat(
                    messages=messages,
                    tools=client.tools_for_model(openai_tools),
                )

            logger.debug(f"Length of chat_history: {len(chat_history)}")

            if client.has_tool_use(chat_response):
                response_text = client.get_response_text(chat_response)
                tool_name, tool_input, tool_use_id = client.get_tool_use(
                    chat_response
                )
                tool_result = process_tool_call(jail, tool_name, tool_input)

                md = Markdown(
                    tool_use_markdown.format(
                        text=response_text,
                        tool_name=tool_name,
                        tool_input=json.dumps(tool_input, indent=2),
                        tool_result="\n".join(
                            tool_result.split("\n")[:5] + ["..."]
                        ),
                    )
                )
                console.print(Panel(md, title="Turn"))

                chat_history.append(
                    client.format_assistant_history_message(chat_response)
                )
                chat_history.append(
                    client.format_tool_result_message(
                        tool_name, tool_use_id, tool_result
                    )
                )
            else:
                chat_history.append(
                    client.format_assistant_history_message(chat_response)
                )
                message_text = (
                    client.get_response_text(chat_response)
                    .replace("<think>", "`<think>`")
                    .replace("</think>", "`</think>`")
                )
                md = Markdown(message_text)
                console.print(Panel(md, title="Code exploration result"))
                if args.output and f:
                    f.write("**Assistant**:\n\n" + message_text)
                break

        # if we're not in chat mode, bail out after first
        # tool use loop.
        if not args.chat:
            break

        # Now the model has finished and is ready for more requests.
        new_prompt = Prompt.ask(
            "Further requests? (leave blank to quit)",
            console=console,
        )
        if new_prompt:
            if args.output and f:
                f.write("**User**:\n\n" + prompt)
            chat_history.append(
                client.format_user_history_message(new_prompt)
            )
        else:
            break

    logger.info("Config: max turns: %d, path: %s", max_turns, jail)
    logger.info("Took %d turns", num_turns)
    logger.info("Used %s model", chat_model)


if __name__ == "__main__":
    main()
