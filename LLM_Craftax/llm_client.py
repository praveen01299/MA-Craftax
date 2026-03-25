import os
import logging
import time
from collections import namedtuple
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

LLMResponse = namedtuple(
    "LLMResponse",
    [
        "model_id",
        "completion",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "reasoning",
    ],
)

class LLMClientWrapper:
    """Wrapper for interacting with the OpenAI API."""

    def __init__(self, config):
        """Initialize the OpenAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        # super().__init__(config)
        self._initialized = False
        self.client_kwargs = {
            "temperature": config.llm_temperature,
            "max_tokens": config.llm_max_tokens,
        }
        self.model_id = config.model_id
        self.max_retries = config.max_retries
        self.delay = config.retry_delay


    def _initialize_client(self):
        """Initialize the OpenAI client if not already initialized."""
        if not self._initialized:
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set in the environment")

            self.client = OpenAI(api_key=OPENAI_API_KEY)

    def execute_with_retries(self, func, *args, **kwargs):
        """Execute a function with retries upon failure.

        Args:
            func (callable): The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The result of the function call.

        Raises:
            Exception: If the function fails after the maximum number of retries.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                logger.error(f"Retryable error during {func.__name__}: {e}. Retry {retries}/{self.max_retries}")
                sleep_time = self.delay * (2 ** (retries - 1))  # Exponential backoff
                time.sleep(sleep_time)
        raise Exception(f"Failed to execute {func.__name__} after {self.max_retries} retries.")

    def convert_messages(self, messages):
        """Convert messages to the format expected by the OpenAI API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the OpenAI API.
        """
        converted_messages = []
        for msg in messages:
            new_content = [{"type": "text", "text": msg.content}]
            # if self.alternate_roles and converted_messages and converted_messages[-1]["role"] == msg.role:
            #     converted_messages[-1]["content"].extend(new_content)
            # else:
            converted_messages.append({"role": msg.role, "content": new_content})
        return converted_messages

    def generate(self, messages):
        """Generate a response from the OpenAI API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the OpenAI API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        # print(converted_messages)
        # exit()

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_completion_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature

            return self.client.chat.completions.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.choices[0].message.content.strip(),
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning=None,
        )
        # return LLMResponse(
        #     model_id="",
        #     completion="NOOP",
        #     stop_reason="",
        #     input_tokens="",
        #     output_tokens="",
        #     reasoning="None",
        # )

