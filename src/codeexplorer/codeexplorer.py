"""
codeexplorer.py

Implements an agent-model to allow LLM to explore a codebase
using tools, rather than trying to pre-create a large context
from the codebase ourselves.
"""

import argparse
import copy
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import anthropic
import ollama
import ibm_watsonx_ai as wai
import ibm_watsonx_ai.foundation_models as waifm
from ibm_watsonx_ai.wml_client_error import WMLClientError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt


logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class OllamaAdapter:
    def __init__(self):
        self.ollama_client = ollama.Client()

    def prepare_messages(self, prompt: str, chat_history):
        messages = [
            {"role": "user", "content": prompt},
        ] + chat_history
        return messages

    def chat(self, **kwargs):
        # message = ollama_client.chat(
        #     model=chat_model,
        #     messages=messages,
        #     tools=openai_tools,
        #     options=ollama.Options(
        #         num_ctx=16384,
        #     ),
        # )
        kwargs["model"] = kwargs["model"] or "qwen3:8b"
        chat_response = self.ollama_client.chat(
            options=ollama.Options(
                num_ctx=16384,
            ),
            **kwargs,
        )
        logger.debug("\nResponse:")
        logger.debug(f"Stop Reason: {chat_response['done_reason']}")
        logger.debug(f"Content: {chat_response['message']}")
        return chat_response

    def get_response_text(self, chat_response) -> str:
        return chat_response.message.content

    def tools_for_model(self, openai_tools):
        return openai_tools

    def has_tool_use(self, chat_response):
        return bool(chat_response.message.tool_calls)

    def get_tool_use(self, chat_response) -> Tuple[str, Dict[str, str], str]:
        tool_call = chat_response["message"]["tool_calls"][0]
        f = tool_call["function"]
        tool_name = f["name"]
        tool_input = f["arguments"]
        return tool_name, tool_input, ""

    def format_assistant_history_message(self, chat_response):
        return {
            "role": "assistant",
            "content": chat_response.message.content,
        }

    def format_tool_result_message(
        self, tool_name: str, tool_use_id: str, tool_result: str
    ) -> Dict:
        return {
            "role": "tool",
            "content": tool_result,
            "name": tool_name,
        }


class AnthropicAdapter:
    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("Set ANTHROPIC_API_KEY")
            raise ValueError("Set ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

    def prepare_messages(self, prompt: str, chat_history):
        # Cache to last prompt to speed up future inference
        cache_prompt = None
        if chat_history:
            cache_prompt = copy.deepcopy(chat_history[-1])
            cache_prompt["content"][0]["cache_control"] = {
                "type": "ephemeral"
            }
        messages = (
            [
                {"role": "user", "content": prompt},
            ]
            + chat_history[:-1]
            + ([cache_prompt] if cache_prompt else [])
        )
        return messages

    def chat(self, **kwargs):
        # message = self.client.messages.create(
        #     model=chat_model,
        #     max_tokens=8192,
        #     messages=cast(Iterable[MessageParam], messages),
        #     tools=cast(Iterable[ToolUnionParam], tools),
        # )
        kwargs["model"] = kwargs["model"] or "claude-3-7-sonnet-latest"
        chat_response = self.client.messages.create(
            max_tokens=8192, **kwargs
        )
        logger.debug("\nResponse:")
        logger.debug(f"Stop Reason: {chat_response.stop_reason}")
        logger.debug(f"Content: {chat_response.content}")
        logger.debug(
            "cache_creation_input_tokens: %d, cache_read_input_tokens: %d, input_tokens: %d",
            chat_response.usage.cache_creation_input_tokens,
            chat_response.usage.cache_read_input_tokens,
            chat_response.usage.input_tokens,
        )
        return chat_response

    def tools_for_model(self, openai_tools):
        """Convert an OpenAI function tools to Anthropic's tool format."""

        def convert_openai_tool_to_anthropic(openai_tool):
            function = openai_tool.get("function", {})

            return {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": function.get("parameters", {}).get(
                        "properties", {}
                    ),
                    "required": function.get("parameters", {}).get(
                        "required", []
                    ),
                },
            }

        return [convert_openai_tool_to_anthropic(x) for x in openai_tools]

    def has_tool_use(self, message):
        return message.stop_reason == "tool_use"

    def get_response_text(self, message) -> str:
        text_block = None
        try:
            text_block = next(
                block for block in message.content if block.type == "text"
            )
        except StopIteration:
            pass  # sometimes the model has nothing to say
        return text_block.text if text_block else ""

    def get_tool_use(self, message) -> Tuple[str, Dict[str, str], str]:
        tool_use = next(
            block for block in message.content if block.type == "tool_use"
        )
        tool_name = tool_use.name
        tool_input = cast(Dict[str, str], tool_use.input)
        return tool_name, tool_input, tool_use.id

    def format_assistant_history_message(self, message):
        return {"role": "assistant", "content": message.content}

    def format_tool_result_message(
        self, tool_name: str, tool_use_id: str, tool_result: str
    ) -> Any:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_result,
                }
            ],
        }


class WatsonxAdapter:
    def __init__(self):
        self.ollama_client = ollama.Client()
        if v := os.environ.get("WATSONX_IAM_API_KEY"):
            wxapikey = v
        else:
            logger.error("WATSONX_IAM_API_KEY")
            raise ValueError("WATSONX_IAM_API_KEY")

        if v := os.environ.get("WATSONX_PROJECT"):
            self.wxproject_id = v
        else:
            logger.error("WATSONX_PROJECT")
            raise ValueError("WATSONX_PROJECT")

        if v := os.environ.get("WATSONX_URL"):
            wxendpoint = v
        else:
            logger.error("WATSONX_URL")
            raise ValueError("WATSONX_URL")

        credentials = wai.Credentials(
            url=wxendpoint,
            api_key=wxapikey,
        )
        self.wxclient = wai.APIClient(credentials)

    def prepare_messages(self, prompt: str, chat_history):
        messages = [
            {"role": "user", "content": prompt},
        ] + chat_history
        return messages

    def chat(self, **kwargs):
        kwargs["model"] = (
            kwargs["model"] or "meta-llama/llama-3-3-70b-instruct"
        )
        params = {
            "time_limit": 30000,
            "max_tokens": 8192,
        }
        wxmodel = waifm.ModelInference(
            model_id=kwargs["model"],
            api_client=self.wxclient,
            params=params,
            project_id=self.wxproject_id,
            space_id=None,
            verify=True,
        )
        chat_response = wxmodel.chat(
            messages=kwargs["messages"],
            tools=kwargs["tools"],
        )
        # logger.debug("\nResponse:")
        # logger.debug(f"Stop Reason: {chat_response['done_reason']}")
        # logger.debug(f"Content: {chat_response['message']}")
        return chat_response

    def get_response_text(self, chat_response) -> str:
        try:
            return chat_response["choices"][0]["message"]["content"]
        except KeyError:
            return ""

    def tools_for_model(self, openai_tools):
        return openai_tools

    def has_tool_use(self, chat_response) -> bool:
        try:
            return (
                len(chat_response["choices"][0]["message"]["tool_calls"]) > 0
            )
        except KeyError:
            return False

    def get_tool_use(self, chat_response) -> Tuple[str, Dict[str, str], str]:
        tool_call = chat_response["choices"][0]["message"]["tool_calls"][0]
        f = tool_call["function"]
        tool_name = f["name"]
        tool_input = json.loads(f["arguments"])
        return tool_name, tool_input, tool_call["id"]

    def format_assistant_history_message(self, chat_response):
        return {
            "role": "assistant",
            "content": self.get_response_text(chat_response),
        }

    def format_tool_result_message(
        self, tool_name: str, tool_use_id: str, tool_result: str
    ) -> Dict:
        return {
            "role": "tool",
            "tool_call_id": tool_use_id,
            "content": tool_result,
        }


#
# Helpers
#


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
    if jail not in p.parents:
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
    root = Path(path).absolute()
    if root != jail and jail not in root.parents:
        return f"ERROR: Path {path} must have {jail} as an ancestor"
    if not root.exists():
        return f"ERROR: Path {root} does not exist"
    if not root.is_dir():
        return f"ERROR: Path {root} is not a directory"

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

    _tree(root)
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

    prompt = f"""
    You are a programmer's assistant exploring a codebase and carrying out programming tasks.

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
    """

    console.print(Panel(Markdown(prompt), title="Prompt"))

    # dedented markdown to use when formatting each tool use message
    tool_use_markdown = """
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
    """

    client: OllamaAdapter | AnthropicAdapter | WatsonxAdapter
    try:
        if args.provider == "ollama":
            client = OllamaAdapter()
        elif args.provider == "anthropic":
            client = AnthropicAdapter()
        elif args.provider == "watsonx":
            client = WatsonxAdapter()
        else:
            raise ValueError("Invalid model provider")
    except Exception:
        logger.error("Cannot load provider")
        exit(1)

    chat_history = []
    num_turns = 0

    for i in range(0, max_turns):
        num_turns += 1
        logger.debug(f"\n{'=' * 50}")

        messages = client.prepare_messages(prompt, chat_history)

        with console.status("Model is working..."):
            logger.debug("Messaging model")
            chat_response = client.chat(
                model=chat_model,
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
            if args.output:
                with open(
                    Path(args.output).absolute(), "w", encoding="utf-8"
                ) as f:
                    f.write(message_text)
            break

    logger.info("Config: max turns: %d, path: %s", max_turns, jail)
    logger.info("Took %d turns", num_turns)
    logger.info("Used %s model", chat_model)


if __name__ == "__main__":
    main()
