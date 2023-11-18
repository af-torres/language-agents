import numpy as np

from ..action import Action
from ..agent import Policy
from ..state import CartPoleState


class Random(Policy):
    def execute(self, _: CartPoleState) -> Action:
        return {
            "decision": np.random.binomial(1, 0.5)
        }