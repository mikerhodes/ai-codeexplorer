from dataclasses import dataclass
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Literal

from codeexplorer import tools
from codeexplorer.adapters import (
    AnthropicAdapter,
    OllamaAdapter,
    WatsonxAdapter,
)

logger = logging.getLogger(__name__)


@dataclass
class AIEvent:
    """AIEvent is a message event for the UX"""

    type: Literal["ai", "getuserinput"]

    # generator fills with ai message when type=ai
    # receiver should print the message
    md: str
    title: str
    message_type: Literal["tooluse", "message", "prompt", "thinking"]

    # for type=getuserinput, this field should be filled in
    # where the yield is received before handing back
    # control to the generator. This was the easiest way
    # to remove the console UX entirely from the AI run loop.
    user_response: str


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

        chat_history.append(
            client.format_assistant_history_message(chat_response)
        )

        # yield the text block for display
        response_thinking = client.get_thinking_text(chat_response)
        if response_thinking:
            yield AIEvent(
                type="ai",
                md=response_thinking,
                title="Thinking",
                message_type="thinking",
                user_response="",
            )
        if output_file:
            output_file.write(
                "**Assistant (thinking)**:\n\n" + response_thinking
            )
        # yield the text block for display
        response_text = client.get_response_text(chat_response)
        yield AIEvent(
            type="ai",
            md=response_text,
            title="Tool use",
            message_type="message",
            user_response="",
        )
        if output_file:
            output_file.write("**Assistant**:\n\n" + response_text)

        # No tool use indicates the end of the "agent loop" and so
        # the AI's turn.
        if not client.has_tool_use(chat_response):
            break

        # yield tool use details
        tool_name, tool_input, tool_use_id = client.get_tool_use(
            chat_response
        )
        tool_result = tools.process_tool_call(jail, tool_name, tool_input)
        chat_history.append(
            client.format_tool_result_message(
                tool_name, tool_use_id, tool_result
            )
        )
        md = TOOL_USE_MARKDOWN.format(
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

    logger.info("Took %d turns", num_turns)


# dedented markdown to use when formatting each tool use message
TOOL_USE_MARKDOWN = textwrap.dedent("""
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
