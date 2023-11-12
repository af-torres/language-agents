import numpy as np

from ..action import Action
from ..agent import Policy
from ..state import CartPoleState


class LLM(Policy):
    def __init__(self) -> None:
        super().__init__()
    
    def execute(self, state: CartPoleState) -> Action:
        return super().execute(state)