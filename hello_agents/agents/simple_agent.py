"""简单的Agent实现，基于OpenAi原生API"""

from typing import Optional, Iterator

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message

class SimpleAgent(Agent):
    """简单的对话Agent"""

    def __init__(
        self,
        name:str,
        llm: HelloAgentsLLM,
        system_prompt:Optional[str] = None,
        config:Optional[Config] = None
    ):
        super().__init__(name, llm, system_prompt, config)

    def run(self, input_text:str, **kwargs) -> str:
        """
        运行简单的我agent
        """
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        for msg in self._history:
            messages.append({'role': msg.role, 'content': msg})

        messages.append({'role': 'user', 'content': input_text})
        response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, 'user'))
        self.add_message(Message(response, 'assistant'))

        return response

    def stream_run(self, input_text:str, **kwargs):
        """流式运行"""
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        for msg in self._history:
            messages.append({'role': msg.role, 'content': msg})

        messages.append({'role': 'user', 'content': input_text})

        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk

        self.add_message(Message(input_text, 'user'))
        self.add_message(Message(full_response, 'assistant'))