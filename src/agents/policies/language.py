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

sysprompt = (
    "You are a control agent responsible to balance a pole in a moving cart. "
    "The pole is attached by an un-actuated joint to the cart, which moves along a frictionless track. "
    "The pendulum is placed upright on the cart. Your objective is to balance the pole by applying forces "
    "in the left and right direction on the cart.\n"
    "The user will inform you of the current cart's position and velocity, and the pole's angle and angular velocity. "
    "You will answer to the user with the action that should be taken in following format: {{decision}}. "
    "Where decision can take the value 0 to push the cart to the left or 1 to push the cart to the right and it should ALWAYS be wrapped by double braces.\n"
    "For example, if your decision is to push the cart to the left, your response to the user should be: {{0}}\n"
    "You will fail the task if the pole angle is greater than ±12° or if the cart position is greater than ±2.4 so make sure that does not happen. But, even if you know you have failed you are still required to respond to the user with a decision.\n"
    "As a final tip, take the following example:\n"
    "user: Cart Position: 1.2\nCart Velocity: 1.7783992290496826\nPole Angle: 3.423493094393619\nPole Angular Velocity: 0.0035346031188965\n"
    "A good strategy could be to try to force the pole angle to go to the left by: 1. moving the cart the left (as many times as necessary) to increase the angular velocity of the pole, 2. move the cart to the right to turn the pole angle to a negative position, 3. slowly move the cart to the left so it stays centered."
    "That is to say, you will want to think on how to move the cart and pole angle to the center.\n"
    "Make sure to add your thinking process to the response to the user, but be concise."
)

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
    def __init__(self, chat: Chat, buffSize: int = 15) -> None:
        self.__chat = chat
        self.__decisionsHistory = []
        self.__buffSize = buffSize
    
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
                    "content": sysprompt
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
            decision = np.random.randint(0,1) # some help to the llm

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