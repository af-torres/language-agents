from abc import ABC, abstractmethod
from .action import Action
from .environment import Environment, Transition, CartPole
from .state import State


class Agent(ABC):
    @abstractmethod
    def action(self, environment: Environment) -> None:
        raise NotImplementedError
    
    def terminate(self) -> None:
        """
        terminate is called once the episode comes to an end
        """
        pass


class Policy(ABC):
    @abstractmethod
    def execute(self, state: State) -> Action:
        raise NotImplementedError
    
    def terminate(self) -> None:
        """
        terminate is called once the episode comes to an end
        """
        pass


class CartPoleAgent(Agent):
    def __init__(self, policy: Policy):
        self.__policy = policy

    def action(self, environment: CartPole) -> Transition:
        currState = environment.state()
        action = self.__policy.execute(currState)
        
        return environment.transition(action)
    
    def terminate(self) -> None:
        self.__policy.terminate()
