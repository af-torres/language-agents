import io
import json
import logging
import re
import time

from typing import Literal

import numpy as np
from ..action import Action
from ..agent import Policy
from ..state import CartPoleState, State
from openai import OpenAI

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Backend = Literal["gpt-4", "gpt-3.5-turbo"]

MINUTE = 60.


class RateLimit(Policy):
    def __init__(self, policy: Policy, rpm: int)-> None:
        self.__executer = policy.execute
        self.__rpm = rpm
        self.__timer = -float("inf")
        self.__count = 0

    def __reset(self) -> None:
        self.__count = 0
        self.__timer = time.perf_counter()
    
    def execute(self, state: State) -> Action:
        diff = time.perf_counter() - self.__timer

        if self.__count >= self.__rpm and diff <= MINUTE:
            time.sleep(MINUTE + 1 - diff)
            self.__reset()
        elif diff > MINUTE:
            self.__reset()
        
        action = self.__executer(state)

        self.__count += 1
        
        return action

class Chat:
    def __init__(self, client: OpenAI, backend: Backend):
        assert backend in Backend.__args__
        self.__backend = backend
        self.__client = client

    def message(self, message):
        return chatCompletionToJson(self.__client.chat.completions.create(
            model=self.__backend,
            messages=message
        ).choices[0].message)

    
class ChatDumpToFile(Chat):
    def __init__(self, client: OpenAI, backend: Backend, f: io.FileIO) -> None:
        super().__init__(client, backend)
        self.__outputs_buff = f
    
    def message(self, message):
        response =  super().message(message)
        self.__outputs_buff.write("Message:\n" + json.dumps(message) + "\n\n" + "Response:\n" + json.dumps(response) + "\n\n")
        
        return response

class LLM(Policy):
    def __init__(self, chat: Chat, systemPrompt: str, buffSize: int = 15) -> None:
        self.__chat = chat
        self.__decisionsHistory = []
        self.__buffSize = buffSize
        self.sysprompt = systemPrompt

    
    def execute(self, state: CartPoleState) -> Action:
        userprompt = [
            {
                "role": "user",
                "content": str(state),
            }
        ]
        response = self.__chat.message(
            [
                {
                    "role": "system",
                    "content": self.sysprompt
                },
            ] + self.__decisionsHistory
            + userprompt
        )

        match = re.search("{{(.*?)}}", response.get("content"))
        try:
            decision = int(match.group().replace("{", "").replace("}", ""))
        except:
            log.info(response)
            log.error("unable to parse a decision from agent")
            decision = np.random.binomial(1, 0.5) # some help to the llm

        self.__decisionsHistory.append(userprompt[0])
        self.__decisionsHistory.append(response)
        self.__decisionsHistory = self.__decisionsHistory[-self.__buffSize:] # dont overflow context window

        return {
            "decision": decision
        }
    
    def terminate(self) -> None:
        self.__decisionsHistory = []

def chatCompletionToJson(message) -> object:
    return {
        "content": message.content,
        "role": message.role
    }