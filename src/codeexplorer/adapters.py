import copy
import json
import logging
import os
import time
from typing import Any, Dict, Tuple, cast

import anthropic
import ibm_watsonx_ai as wai
import ibm_watsonx_ai.foundation_models as waifm
import ollama

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class OllamaAdapter:
    def __init__(self, model: str):
        self.model = model if model else "qwen3:8b"
        self.ollama_client = ollama.Client()

    def prepare_messages(self, prompt: str, chat_history):
        messages = [
            {"role": "user", "content": prompt},
        ] + chat_history
        return messages

    def chat(self, messages, tools):
        # message = ollama_client.chat(
        #     model=chat_model,
        #     messages=messages,
        #     tools=openai_tools,
        #     options=ollama.Options(
        #         num_ctx=16384,
        #     ),
        # )
        chat_response = self.ollama_client.chat(
            model=self.model,
            options=ollama.Options(
                num_ctx=16384,
            ),
            messages=messages,
            tools=tools,
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
    def __init__(self, model: str):
        self.model = model if model else "claude-sonnet-4-20250514"
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

    def chat(self, messages, tools):
        retries = 5
        chat_response = None
        while retries:
            retries -= 1
            try:
                chat_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    messages=messages,
                    tools=tools,
                    thinking={"type": "enabled", "budget_tokens": 4096},
                )
            except anthropic.RateLimitError as ex:
                if retries:
                    logger.warning(
                        "Rate limit error; retry in 65s: %s", ex.message
                    )
                    time.sleep(65)
                else:
                    logger.error("Too many 429s, giving up: %s", ex.message)
        if not chat_response:
            return None

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
    def __init__(self, model: str):
        self.model = model if model else "meta-llama/llama-3-3-70b-instruct"
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
        self.wxmodel = waifm.ModelInference(
            model_id=self.model,
            api_client=wai.APIClient(credentials),
            project_id=self.wxproject_id,
            space_id=None,
            verify=True,
            params={
                "time_limit": 30000,
                "max_tokens": 8192,
            },
        )

    def prepare_messages(self, prompt: str, chat_history):
        messages = [
            {"role": "user", "content": prompt},
        ] + chat_history
        return messages

    def chat(self, messages, tools):
        chat_response = self.wxmodel.chat(
            messages=messages,
            tools=tools,
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
