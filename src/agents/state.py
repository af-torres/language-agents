import math

from abc import ABC
from typing import Tuple


class Base(ABC):
    @property
    def name(self)-> str: 
        raise NotImplementedError


class CartPoleState(Base):
    def __init__(self, observation: Tuple[float, float, float, float]):
        self.cartPosition = observation[0]
        self.cartVelocity = observation[1]
        self.poleAngle = observation[2] * 180 / math.pi
        self.poleAngularVelocity = observation[3]

    def name(self) -> str:
        return "Cart Pole State"
    
    def CartPosition(self) -> float:
        return self.cartPosition
    
    def CartVelocity(self) -> float:
        return self.cartVelocity
    
    def PoleAngle(self) -> float:
        return self.poleAngle
    
    def PoleAngularVelocity(self) -> float:
        return self.poleAngularVelocity
    
    def __str__(self) -> str:
        return f"""Cart Position: {self.CartPosition()}
Cart Velocity: {self.CartVelocity()}
Pole Angle: {self.PoleAngle()}
Pole Angular Velocity: {self.PoleAngularVelocity()}"""


State = CartPoleState
