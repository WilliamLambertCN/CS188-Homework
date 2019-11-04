# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        # print("new_food:", newFood)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFood = newFood.asList()  # list
        ghostPos = []
        for G in newGhostStates:
            ghostPos_ = G.getPosition()[0], G.getPosition()[1]
            ghostPos.append(ghostPos_)
        # ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
        scared = newScaredTimes[0] > 0
        # if not new ScaredTimes new state is ghost: return lowest value
        if not scared and (newPos in ghostPos):
            return -1.0

        if newPos in currentGameState.getFood().asList():
            return 1

        closestFoodDist = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
        closestGhostDist = sorted(ghostPos, key=lambda gDist: util.manhattanDistance(gDist, newPos))

        fd = lambda fDis: util.manhattanDistance(fDis, newPos)

        gd = lambda gDis: util.manhattanDistance(gDis, newPos)

        return 1 / fd(closestFoodDist[0]) - 1 / gd(closestGhostDist[0])
        "*** YOUR CODE HERE ***"

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        GhostIndex = [i for i in range(1, gameState.getNumAgents())]

        def term(state, d):
            return state.isWin() or state.isLose() or d == self.depth

        def min_value(state, d, ghost):  # minimizer

            if term(state, d):
                return self.evaluationFunction(state)

            v = 10000000000000000
            for action in state.getLegalActions(ghost):
                if ghost == GhostIndex[-1]:
                    v = min(v, max_value(state.generateSuccessor(ghost, action), d + 1))
                else:
                    v = min(v, min_value(state.generateSuccessor(ghost, action), d, ghost + 1))
            # print(v)
            return v

        def max_value(state, d):  # maximizer

            if term(state, d):
                return self.evaluationFunction(state)

            v = -10000000000000000
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), d, 1))
            # print(v)
            return v

        res = [(action, min_value(gameState.generateSuccessor(0, action), 0, 1)) for action in
               gameState.getLegalActions(0)]
        res.sort(key=lambda k: k[1])

        return res[-1][0]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        now_value = -1e10
        alpha = -1e10
        beta = 1e10
        next_PacmanAction = Directions.STOP

        legal_actions = gameState.getLegalActions(0).copy()

        for next_action in legal_actions:
            nextState = gameState.generateSuccessor(0, next_action)

            next_value = self.get_node_value(nextState, 0, 1, alpha, beta)
            # same as v = max(v, value(successor))
            if next_value > now_value:
                now_value, next_PacmanAction = next_value, next_action
            alpha = max(alpha, now_value)
        return next_PacmanAction
        util.raiseNotDefined()

    def get_node_value(self, gameState, cur_depth=0, agent_index=0, alpha=-1e10, beta=1e10):
        """
        Using self-defined function, alpha_value(), beta_value() to choose the most appropriate action
        Only when it's the final state, can we get the value of each node, using the self.evaluationFunction(gameState)
        Otherwise we just get the alpha/beta value we defined here.
        """
        max_party = [0, ]
        min_party = list(range(1, gameState.getNumAgents()))

        if cur_depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        elif agent_index in max_party:
            return self.alpha_value(gameState, cur_depth, agent_index, alpha, beta)
        elif agent_index in min_party:
            return self.beta_value(gameState, cur_depth, agent_index, alpha, beta)
        else:
            print('Errors occur in your party division !!! ')

    def alpha_value(self, gameState, cur_depth, agent_index, alpha=-1e10, beta=1e10):
        v = -1e10
        legal_actions = gameState.getLegalActions(agent_index)
        for index, action in enumerate(legal_actions):
            next_v = self.get_node_value(gameState.generateSuccessor(agent_index, action),
                                         cur_depth, agent_index + 1, alpha, beta)
            v = max(v, next_v)
            if v > beta:  # next_agent in which party
                return v
            alpha = max(alpha, v)
            # print("alpha>> ", alpha)
        return v

    def beta_value(self, gameState, cur_depth, agent_index, alpha=-1e10, beta=1e10):
        """
        min_party, search for minimums
        """
        v = 1e10
        legal_actions = gameState.getLegalActions(agent_index)
        for index, action in enumerate(legal_actions):
            if agent_index == gameState.getNumAgents() - 1:
                next_v = self.get_node_value(gameState.generateSuccessor(agent_index, action),
                                             cur_depth + 1, 0, alpha, beta)
                v = min(v, next_v)  # begin next depth
                if v < alpha:
                    # print("pruning in beta_value")
                    return v
            else:
                next_v = self.get_node_value(gameState.generateSuccessor(agent_index, action),
                                             cur_depth, agent_index + 1, alpha, beta)
                v = min(v, next_v)  # begin next depth
                if v < alpha:  # next agent goes on at the same depth
                    # print("pruning in beta_value")
                    return v
            beta = min(beta, v)
            # print("beta>> ", beta)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    INF = 100000.0

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        maxValue = -self.INF
        maxAction = Directions.STOP

        for action in gameState.getLegalActions(agentIndex=0):
            sucState = gameState.generateSuccessor(action=action, agentIndex=0)
            sucValue = self.expNode(sucState, currentDepth=0, agentIndex=1)
            if sucValue > maxValue:
                maxValue = sucValue
                maxAction = action

        return maxAction

    def maxNode(self, gameState, currentDepth):
        if currentDepth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        maxValue = -self.INF
        for action in gameState.getLegalActions(agentIndex=0):
            sucState = gameState.generateSuccessor(action=action, agentIndex=0)
            sucValue = self.expNode(sucState, currentDepth=currentDepth, agentIndex=1)
            if sucValue > maxValue:
                maxValue = sucValue
        return maxValue

    def expNode(self, gameState, currentDepth, agentIndex):
        if currentDepth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        numAction = len(gameState.getLegalActions(agentIndex=agentIndex))
        totalValue = 0.0
        numAgent = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex=agentIndex):
            sucState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            if agentIndex == numAgent - 1:
                sucValue = self.maxNode(sucState, currentDepth=currentDepth + 1)
            else:
                sucValue = self.expNode(sucState, currentDepth=currentDepth, agentIndex=agentIndex + 1)
            totalValue += sucValue

        return totalValue / numAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # Consts
    INF = 100000000.0  # Infinite value
    WEIGHT_FOOD = 10.0  # Food base value
    WEIGHT_GHOST = -10.0  # Ghost base value
    WEIGHT_SCARED_GHOST = 100.0  # Scared ghost base value

    # Base on gameState.getScore()
    score = currentGameState.getScore()

    # Evaluate the distance to the closest food
    distancesToFoodList = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    if len(distancesToFoodList) > 0:
        score += WEIGHT_FOOD / min(distancesToFoodList)
    else:
        score += WEIGHT_FOOD

    # Evaluate the distance to ghosts
    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:  # If scared, add points
                score += WEIGHT_SCARED_GHOST / distance
            else:  # If not, decrease points
                score += WEIGHT_GHOST / distance
        else:
            return -INF  # Pacman is dead at this point

    return score

# Abbreviation
better = betterEvaluationFunction