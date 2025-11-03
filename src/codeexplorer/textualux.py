from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, cast

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Input,
    TextArea,
)
from textual.widgets import Markdown as TextualMarkdown

from codeexplorer import turn
from codeexplorer import prompt
from codeexplorer.adapters import (
    AnthropicAdapter,
    OllamaAdapter,
    WatsonxAdapter,
)


class AICodeExplorer(App):
    CSS_PATH = "./textual.tcss"
    chat_history = []
    first_input = True

    BINDINGS = [
        Binding("ctrl+e", "submit_prompt", "Submit Prompt", priority=True),
        Binding("ctrl+u", "page_up", "Page Up", priority=True),
        Binding("ctrl+d", "page_down", "Page Down", priority=True),
        Binding("ctrl+w", "write_out", "Write Out", priority=True),
        # Binding("up", "scroll_up", "Scroll Up", priority=True),
        # Binding("down", "scroll_down", "Scroll Down", priority=True),
    ]

    def __init__(
        self,
        initial_task: str,
        openai_tools: List[Dict[str, Any]],
        client: OllamaAdapter | AnthropicAdapter | WatsonxAdapter,
        output_file,
        max_turns: int,
        jail: Path,
    ):
        self.initial_task = initial_task
        self.openai_tools = openai_tools
        self.client = client
        self.output_file = output_file
        self.max_turns = max_turns
        self.jail = jail
        self.prompt = prompt.get_prompt("", jail)
        self.SUB_TITLE = str(jail)
        super().__init__()

    async def on_mount(self) -> None:
        if self.initial_task:
            await self.new_user_message(self.initial_task)

    @dataclass
    class MarkdownEvent(Message):
        md: str

    @dataclass
    class ToolUseEvent(Message):
        md: str
        title: str

    @dataclass
    class ThinkingEvent(Message):
        md: str
        title: str

    class AITurnDoneEvent(Message):
        pass

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield Collapsible(
                TextualMarkdown(self.prompt),
                title="System Prompt",
            )
        with VerticalScroll(id="chat-input"):
            yield TextArea(
                text=self.initial_task or "",
                id="chatbox",
            )
        # yield Button("Stop", id="progress")
        yield Footer()

    @on(Input.Submitted)
    async def on_input(self, event: Input.Submitted) -> None:
        if event.value == "/exit":
            self.exit(0)
            return
        await self.new_user_message(event.value)

    def action_page_up(self) -> None:
        self.query_one("#chat-view").scroll_relative(y=-3)

    def action_page_down(self) -> None:
        self.query_one("#chat-view").scroll_relative(y=3)

    async def action_submit_prompt(self) -> None:
        text: str = cast(TextArea, self.query_one("#chatbox")).text
        if not text:
            return
        if text == "/exit":
            self.exit(0)
            return
        await self.new_user_message(text)

    def action_write_out(self) -> None:
        with open("chat.md", "w") as f:
            for x in self.chat_history:
                import json

                f.write(json.dumps(x))
                f.write("\n\n")

    async def new_user_message(self, message: str):
        """Process a new user message"""
        cv = self.query_one("#chat-view")
        md = TextualMarkdown(message, classes="prompt")
        md.border_title = "User"
        await cv.mount(md)
        cv.scroll_end()
        self.chat_history.append(
            self.client.format_user_history_message(message)
        )
        cb = cast(Input, self.query_one("#chatbox"))
        cb.loading = True
        cb.clear()
        self.process_with_ai(self.prompt)

    @work(thread=True)
    def process_with_ai(self, system_prompt: str) -> None:
        self.log("Chat history length:", len(self.chat_history))
        try:
            msgs = turn.run_ai_turn(
                system_prompt,
                self.openai_tools,
                self.client,
                self.output_file,
                self.max_turns,
                self.jail,
                self.chat_history,
            )
            while True:
                ai_event = next(msgs)
                if ai_event.type == "ai":
                    match ai_event.message_type:
                        case "message":
                            self.post_message(
                                self.MarkdownEvent(ai_event.md)
                            )
                        case "tooluse":
                            self.post_message(
                                self.ToolUseEvent(
                                    ai_event.md, ai_event.title
                                )
                            )
                        case "thinking":
                            self.post_message(
                                self.ThinkingEvent(
                                    ai_event.md, ai_event.title
                                )
                            )
                        case "prompt":
                            self.post_message(
                                self.MarkdownEvent(ai_event.md)
                            )

        except StopIteration:
            self.log("StopIteration")
            pass

        self.post_message(self.AITurnDoneEvent())

    def on_aicode_explorer_aiturn_done_event(self):
        """Respond to AITurnDoneEvent messages"""
        cb = self.query_one("#chatbox")
        cb.loading = False
        cb.focus()

    async def on_aicode_explorer_markdown_event(
        self, message: MarkdownEvent
    ):
        """Respond to MarkdownEvent messages"""
        widget = TextualMarkdown(message.md)
        cv = self.query_one("#chat-view")
        await cv.mount(widget)
        # scroll top of message into view, ready for reading
        cv.scroll_to_widget(widget)

    async def on_aicode_explorer_tool_use_event(self, message: ToolUseEvent):
        """Respond to ToolUseEvent messages"""
        widget = Collapsible(
            TextualMarkdown(message.md),
            title=message.title,
        )
        cv = self.query_one("#chat-view")
        await cv.mount(widget)
        cv.scroll_end()

    async def on_aicode_explorer_thinking_event(
        self, message: ThinkingEvent
    ):
        """Respond to ThinkingEvent messages"""
        widget = Collapsible(
            TextualMarkdown(message.md), title=message.title, collapsed=False
        )
        cv = self.query_one("#chat-view")
        await cv.mount(widget)
        cv.scroll_end()
