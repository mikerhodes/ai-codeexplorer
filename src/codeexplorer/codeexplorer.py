"""
codeexplorer.py

Implements an agent-model to allow LLM to explore a codebase
using tools, rather than trying to pre-create a large context
from the codebase ourselves.
"""

import argparse
from dataclasses import dataclass
import json
import logging
import signal
import subprocess
import textwrap
from pathlib import Path
from typing import Callable, Dict, List, Any, Literal

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from codeexplorer.adapters import (
    AnthropicAdapter,
    OllamaAdapter,
    WatsonxAdapter,
)
from codeexplorer import tools


@dataclass
class AIEvent:
    """AIEvent is a message event for the UX"""

    type: Literal["ai", "getuserinput"]

    # generator fills with ai message when type=ai
    # receiver should print the message
    md: str
    title: str
    message_type: Literal["tooluse", "message", "prompt"]

    # for type=getuserinput, this field should be filled in
    # where the yield is received before handing back
    # control to the generator. This was the easiest way
    # to remove the console UX entirely from the AI run loop.
    user_response: str


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

    active_tools = tools.tools
    if args.allow_edits:
        if not is_git_working_copy_clean(Path(args.path)):
            print("To edit, git working directory must be clean:", args.path)
            exit(1)
        active_tools.extend(tools.edit_tools)

    openai_tools = [
        {"type": "function", "function": x} for x in active_tools
    ]

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

    f = None
    if args.output:
        f = open(Path(args.output).absolute(), "w", encoding="utf-8")
    if args.output and f is None:
        logger.error("Could not open output file; exiting.")
        exit(1)

    run_rich_ux(
        args.chat,
        args.task,
        openai_tools,
        client,
        f,
        max_turns,
        jail,
    )

    logger.info("Config: max turns: %d, path: %s", max_turns, jail)
    logger.info("Used %s model", chat_model)


def run_rich_ux(
    chat_mode: bool,
    user_task: str,
    openai_tools: List[Dict[str, Any]],
    client: OllamaAdapter | AnthropicAdapter | WatsonxAdapter,
    output_file,
    max_turns: int,
    jail: Path,
):
    """Run CodeExplorer's rich-based UX"""
    console = Console()

    # Allow the UX loop to request user input
    def request_user_input(prompt: str) -> str:
        return Prompt.ask(
            prompt,
            console=console,
        )

    try:
        msgs = run_ux_loop(
            chat_mode,
            user_task,
            openai_tools,
            client,
            output_file,
            max_turns,
            jail,
            request_user_input,
        )
        while True:
            with console.status("Model is working..."):
                ai_event = next(msgs)
            if ai_event.type == "ai":
                match ai_event.message_type:
                    case "message":
                        btype = box.SIMPLE
                    case "tooluse":
                        btype = box.ROUNDED
                    case "prompt":
                        btype = box.ROUNDED
                console.print(
                    Panel(
                        Markdown(ai_event.md),
                        title=ai_event.title,
                        box=btype,
                    )
                )
            elif ai_event.type == "getuserinput":
                ai_event.user_response = request_user_input(ai_event.title)

    except StopIteration:
        pass


def run_ux_loop(
    chat_mode: bool,
    user_task: str,
    openai_tools: List[Dict[str, Any]],
    client: OllamaAdapter | AnthropicAdapter | WatsonxAdapter,
    output_file,
    max_turns: int,
    jail: Path,
    request_user_input: Callable[[str], str],
):
    # Generate and print initial prompt
    if not user_task:
        e = AIEvent(
            type="getuserinput",
            message_type="message",
            title="Anything you'd like AI to think about (eg, plan how to do X)",
            md="",
            user_response="",
        )
        yield e
        extras = e.user_response
        if extras:
            user_task = extras
        else:
            user_task = "Please explain this codebase"
    prompt = get_prompt(user_task, jail)
    yield AIEvent(
        type="ai",
        md=prompt,
        title="Prompt",
        message_type="prompt",
        user_response="",
    )

    chat_history = []

    while True:
        # Run agent loop for this turn
        for ev in run_ai_turn(
            prompt,
            openai_tools,
            client,
            output_file,
            max_turns,
            jail,
            chat_history,
        ):
            yield ev

        # If we're not in chat mode, bail out after first
        # tool use loop.
        if not chat_mode:
            break

        # Now the model has finished and is ready for more requests.
        e = AIEvent(
            type="getuserinput",
            message_type="message",
            title="Further requests? (leave blank to quit)",
            md="",
            user_response="",
        )
        yield e
        new_user_request = e.user_response
        if new_user_request:
            if output_file:
                output_file.write("**User**:\n\n" + prompt)
            chat_history.append(
                client.format_user_history_message(new_user_request)
            )
        else:
            break


def run_ai_turn(
    prompt: str,
    openai_tools: List[Dict[str, Any]],
    client: OllamaAdapter | AnthropicAdapter | WatsonxAdapter,
    output_file,
    max_turns: int,
    jail: Path,
    chat_history: List,
):
    num_turns = 0

    # The "agent loop" --- loop until the model stops
    # requesting tools
    for _ in range(0, max_turns):
        num_turns += 1
        logger.debug(f"\n{'=' * 50}")

        messages = client.prepare_messages(prompt, chat_history)

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
            tool_result = tools.process_tool_call(
                jail, tool_name, tool_input
            )

            md = TOOL_USE_MARKDOWN.format(
                text=response_text,
                tool_name=tool_name,
                tool_input=json.dumps(tool_input, indent=2),
                tool_result="\n".join(tool_result.split("\n")[:5] + ["..."]),
            )
            yield AIEvent(
                type="ai",
                md=md,
                title="Tool use - {}".format(tool_name),
                message_type="tooluse",
                user_response="",
            )

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
            yield AIEvent(
                type="ai",
                md=message_text,
                title="Code exploration result",
                message_type="message",
                user_response="",
            )
            if output_file:
                output_file.write("**Assistant**:\n\n" + message_text)
            break

    logger.info("Took %d turns", num_turns)


# dedented markdown to use when formatting each tool use message
TOOL_USE_MARKDOWN = textwrap.dedent("""
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


def get_prompt(user_task: str, jail: Path) -> str:
    # Prompt notes
    # Hard-coding paragraph comes from Sonnet 4 system card.
    # (via https://simonwillison.net/2025/May/25/claude-4-system-card/)
    return textwrap.dedent(f"""
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

    The project root directory is: {jail.resolve()}

    Here's the user's task:

    {user_task}
    """)


if __name__ == "__main__":
    main()
