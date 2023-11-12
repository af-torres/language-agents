import gymnasium as gym

from abc import ABC, abstractmethod
from typing import Tuple, Type
from .action import Action
from .state import State, CartPoleState


Reward = float | int
Transition = Tuple[Type["Environment"], Reward]


class Environment(ABC):
    @abstractmethod
    def transition(self, action: Action) -> Transition:
        raise NotImplementedError
    
    @abstractmethod
    def state(self) -> State:
        raise NotImplementedError
    
    @abstractmethod
    def terminated(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def render(self) -> None:
        raise NotImplementedError
    

class CartPole(Environment):
    def __init__(self, render_mode: str | None = None):
        self.__env = gym.make(
            "CartPole-v1",
            render_mode=render_mode,
        )
        self.__state = CartPoleState(self.__env.reset()[0])
        self.__terminated = False

    def transition(self, action: Action) -> Transition:
        a = action["decision"]
        if not isinstance(a, int):
            raise ValueError(         
    """Decision can take a value in {0, 1} indicating the direction the cart is pushed with.
    + 0: Push cart to the left
    + 1: Push cart to the right""")
        
        step = self.__env.step(a)
        self.__state = CartPoleState(step[0])
        self.__terminated = step[2]
        
        return [self, step[1]] 
    
    def state(self) -> CartPoleState:
        return self.__state
    
    def random_state(self) -> None:
        sample = self.__env.observation_space.sample()
        self.__env.state = sample
        self.__state = CartPoleState(sample)
    
    def terminated(self) -> bool:
        return self.__terminated
    
    def close(self) -> None:
        self.__env.close()

    def reset(self) -> None:
        self.__env.reset()
        self.__terminated = False

    def render(self) -> None:
        self.__env.render()
