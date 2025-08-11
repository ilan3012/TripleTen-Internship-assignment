"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined computer use tools.
"""

import platform
from collections.abc import Callable
from datetime import datetime
import json
from enum import StrEnum
from typing import Any, cast

from openai import OpenAI, APIError

from .tools import (
    TOOL_GROUPS_BY_VERSION,
    ToolCollection,
    ToolResult,
    ToolVersion,
)


class APIProvider(StrEnum):
    Nebius = "nebius"



# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_based_edit_tool or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.

here is a list of tools you can use:
* **bash**: Use this tool to run bash commands on the virtual machine. You can use it to install applications, run scripts, and perform other tasks that require command line access.
* **computer**: Use this tool to interact with the virtual machine's desktop environment. You can use it to open applications, take screenshots, and perform other tasks that require a graphical user interface.
* **edit**: Use this tool to edit text files on the virtual machine. You can use it to create, modify, and delete files, as well as to perform other text editing tasks.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* when passing arguments to tools if they are a number or a boolean, pass them as a number or boolean, not as a string.
* when passing arguments to tools, if they are a list, pass them as a list, not as a string.
</IMPORTANT>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your str_replace_based_edit_tool.
</IMPORTANT>""" 


async def sampling_loop(
    *,
    model: str,
    system_prompt_suffix: str,
    messages: list[dict[str, Any]],
    output_callback: Callable[[str | ToolResult], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    _api_error_callback: Callable[
        [APIError], None
    ],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    tool_version: ToolVersion,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))
    system = f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"

    while True:
        image_truncation_threshold = only_n_most_recent_images or 0
        client = OpenAI(
            base_url = "https://api.studio.nebius.com/v1/",
            api_key = api_key)

        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                messages,
                only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )


        # Call the API
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.
        try:
            messages[0]["content"] = system #set system prompt
            raw_response = client.chat.completions.create(
                max_completion_tokens=max_tokens,
                messages=messages,
                model=model,
                tools=tool_collection.to_params())
        except APIError as e:
            _api_error_callback(e)
            return messages

        response = json.loads(raw_response.to_json())

        response_params = response["choices"][0]
        

        if response_params["finish_reason"] == "stop":
            content = response_params["message"]["content"]
            output_callback(content)
            messages.append({"content": content, "role": "assistant"})
            return messages
        elif response_params["finish_reason"] == "tool_calls":
            tools = response_params["message"]["tool_calls"]
            tool_result_content: list[Any] = []
            output_callback(response_params["message"]["content"])
            for tool in tools:
                result = await tool_collection.run(
                        name=tool["function"]["name"],
                        tool_input=cast(dict[str, Any], json.loads(tool["function"]["arguments"])),
                    )
                api_tool_result = _make_api_tool_result(result, tool["id"])
                tool_result_content.append(
                        api_tool_result
                    )
                tool_output_callback(result, tool["id"])
                messages.append({"content": api_tool_result["content"], "role": "tool", "tool_call_id": tool["id"]})


def _maybe_filter_to_n_most_recent_images(
    messages: list[dict[str, Any]],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ]

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> dict[str, Any]:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[dict[str, Any]] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
