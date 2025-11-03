"""
codeexplorer.py

Implements an agent-model to allow LLM to explore a codebase
using tools, rather than trying to pre-create a large context
from the codebase ourselves.
"""

import argparse
import logging
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from codeexplorer import tools
from codeexplorer.adapters import (
    AnthropicAdapter,
    OllamaAdapter,
    WatsonxAdapter,
)
from codeexplorer import textualux


logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def signal_handler(sig, frame):
    print("Interrupted; exiting.")
    exit(0)


signal.signal(signal.SIGINT, signal_handler)


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

    # Let model explore this folder for now
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

    run_textual_ux(
        args.task,
        openai_tools,
        client,
        f,
        max_turns,
        jail,
    )

    logger.info("Config: max turns: %d, path: %s", max_turns, jail)
    logger.info("Used %s model", chat_model)


def run_textual_ux(
    user_task: str,
    openai_tools: List[Dict[str, Any]],
    client: OllamaAdapter | AnthropicAdapter | WatsonxAdapter,
    output_file,
    max_turns: int,
    jail: Path,
):
    """Run CodeExplorer's rich-based UX"""

    app = textualux.AICodeExplorer(
        initial_task=user_task,
        openai_tools=openai_tools,
        client=client,
        output_file=output_file,
        max_turns=max_turns,
        jail=jail,
    )
    app.run()


if __name__ == "__main__":
    main()
