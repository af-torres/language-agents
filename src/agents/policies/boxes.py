import numpy as np

from ..action import Action
from ..agent import Policy
from ..state import CartPoleState


class Boxes(Policy):
    """
    Boxes implements the policy for the Cart Pole classic control problem described in
    https://citeseerx.ist.psu.edu/document?doi=2f027193fb703d0af58ec382bd1438daff9417d7
    
    The main idea of this policy is to divide the environment into mutually
    independent sub-games that can be mapped from the environment states
    and learn a policy for each of the sub-games. For example, when the pole
    angle is between (x1,x2), the pole angle velocity is between (x3,x4), 
    the cart position is between (x5,x6), and the cart velocity is between (x7,x8)
    then use policy p. Where p is either a deterministic or probabilistic function
    called a 'genie' or 'demon' that knows what action is best for the current state.

    In the paper, the 'model' was constructed by quantising the state variables by setting
    thresholds on the four measurement scales. Here, we will use the notion of a quantile
    to refer to this thresholds and will adapt the threshold to open AI's gym environment
    observation's range so that relative distance between quantiles is maintained.
    """


    def __init__(self, dk=.99, k=20, c0=0, c1=1):
        """
        Let X1 = cart position, X2 = cart velocity, X3 = pole angle, X4 = angle velocity,
        State = 4-Tuple(X1,X2,X3,X4), Q: State -> 4-Tuple(N), QS: Xi -> N, and
        G: State -> Genie. Where Q is a function that computes the Quantile of every
        State variable, QS is a function that computes the number of quantiles a state
        variable has, and G is a function that maps from a state to  the Genie responsible 
        for playing the State's sub-game.

        During initialization of this class, we want to create the Genies and set up
        variables that make G easy to implement by the __genie method. For that, we create
        an array of n = QS(X1) * QS(X2) * ... * QS(X4) Genies and reshape it into an array of
        shape (QS(X1), QS(X2), ..., QS(X4), 1). With this array, we can find the genie
        responsible for making the decision in the current state by computing the Q(State)
        and using the result for array indexing.

        For example, say the current state = (-5, 1, 8, 1) and the corresponding quantiles
        for each of the state variables are (1, 1, 2, 2). Then the Genie responsible for 
        playing the current state sub-game would be the genie at index (0,0,1,1,0).

        dk, k, c0, c1 are hyperparameters for which the default value is the one found by
        the original paper authors.
        """

        self.dk, self.k, self.c0, self.c1 = dk, k, c0, c1
        self.globalLive, self.globalUsage = 0, 0

        self.cartPositionQuantiles = np.array([-2.4, -1.44, -0.48, 0.48, 1.44, 2.4, float("inf")])
        self.cartVelocityQuantiles = np.array([-30, -1, 1, 30, float("inf")])
        self.poleAngleQuantiles = np.array([-12, -6, -1, 1, 6, 12, float("inf")])
        self.poleAngularVelocityQuantiles = np.array([-24, -6, 6, 24, float("inf")])
        
        quantiles = [
            self.cartPositionQuantiles,
            self.cartVelocityQuantiles,
            self.poleAngleQuantiles,
            self.poleAngularVelocityQuantiles,
        ]
        n = 1 # number of genies
        shape = [] # final shape of genies array
        for q in quantiles:
            s = q.shape[0]
            n *= s
            shape.append(s)
    
        shape.append(1)
        
        self.genies = np.array(
            [genie(int(s)) for s in np.random.binomial(1, .5, n)]
        ).reshape(tuple(shape))
        self.n = n
        self.t = 0

    def __genie(self, state: CartPoleState, t) -> Action:
        idx = [
            np.searchsorted(self.cartPositionQuantiles, state.CartPosition()),
            np.searchsorted(self.cartVelocityQuantiles, state.CartVelocity()),
            np.searchsorted(self.poleAngleQuantiles, state.PoleAngle()),
            np.searchsorted(self.poleAngularVelocityQuantiles, state.PoleAngularVelocity()),
            0
        ]

        return self.genies[tuple(idx)].action(t)

    def execute(self, state: CartPoleState) -> Action:
        currTime = self.t
        self.t += 1
        return self.__genie(state, currTime)

    def terminate(self) -> None:
        endTime = self.t
        self.t = 1
        self.globalLive = self.globalLive * self.dk + endTime
        self.globalUsage = self.globalUsage * self.dk + 1

        merit = self.globalLive / self.globalUsage
        target = self.c0 + self.c1 * merit

        g: genie
        for g in self.genies.reshape((self.n,)):
            g.updatePolicy(endTime, target, self.dk, self.k)


class genie:
    def __init__(self, s: int):
        """
        Each genie starts with a random state decision s. Games are played over time
        and the decision is updated based on the criteria described in the paper.
        """

        self.leftLife, self.rightLife = 0, 0
        self.leftUsage, self.rightUsage = 0, 0
        
        self.decisionTime = np.array([])

        self.s = s # 0 means left, 1 means right

    def action(self, t) -> Action:
        self.decisionTime = np.append(self.decisionTime, t)
        return {"decision": self.s}
    
    def updatePolicy(self, endTime: int, target: float, dk: float, k: float) -> None:
        if len(self.decisionTime) == 0:
            return

        if self.s == 0:
            self.leftLife = self.leftLife * dk + np.sum(endTime - self.decisionTime)
            self.leftUsage = self.leftUsage * dk + len(self.decisionTime)

            self.rightLife = self.rightLife * dk
            self.rightUsage = self.rightUsage * dk
        else:
            self.rightLife = self.rightLife * dk + np.sum(endTime - self.decisionTime)
            self.rightUsage = self.rightUsage * dk + len(self.decisionTime)

            self.leftLife = self.leftLife * dk
            self.leftUsage = self.leftUsage * dk

        leftValue = (self.leftLife + k * target) / (self.leftUsage + k)
        rightValue = (self.rightLife + k * target) / (self.rightUsage + k)

        if leftValue > rightValue:
            self.s = 0
        else:
            self.s = 1

        self.decisionTime = np.array([])
