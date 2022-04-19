# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    self.vals = util.Counter()

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
    if (state,action) in self.vals:
        return self.vals[(state, action)]
    else:
        return 0.0

  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    if not self.getLegalActions(state):
      return 0.0
    
    q_vals = [self.getQValue(state,action) for action in self.getLegalActions]
    max_action = max(q_vals)
    
    return max_action

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    if not self.getLegalActions(state):
      return None
    
    mx_val = -10000
    req_act = None 
    for i in self.getLegalActions(state):
      temp_q = self.getQValue(state,i)
      if (mx_val <  temp_q):
        mx_val = temp_q
        req_act = i
    
    return req_act
    
    util.raiseNotDefined()

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
    if not legalActions:
        return action

    if util.flipCoin(self.epsilon):
        action = (random.choice(legalActions))
        print("IF",action)
    else:
        action = (self.getPolicy(state))
        print("ELSE",action)
    
    print(action)
    return action
    util.raiseNotDefined()

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    """Q(s,a) = (1-alpha)Q(s,a)+alpha[r + gamma*max(Q(s',a')-Q(s,a))]"""
    reward = 0.1
    temp = self.vals[(state,action)] 
    nextActions = self.getLegalActions(nextState)
    if not nextActions:
      self.vals[(nextState,action)] = 0
    else:
      print("HERE CALCULATING")
      mx = max([self.getQValue(nextState,act) for act in nextActions])
      print(mx)
      quantity = reward + self.discount*mx
      print(quantity)
      self.vals[(nextState,action)] = self.getQValue(state,action)+self.alpha*(quantity+self.getQValue(state,action))
    

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weights = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    Q = 0
    featureVector = self.featExtractor.getFeatures(state, action)
    for i in featureVector:
        Q += self.weights[i] * featureVector[i]
    return Q
    util.raiseNotDefined()

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    features = self.featExtractor.getFeatures(state, action)
    i = 0
    legalActions = self.getLegalActions(nextState)

    for feat in features:
      diff = 0
      if not legalActions:
          diff = reward - self.getQValue(state, action)
      else:
          diff = (reward + self.discount * max([self.getQValue(nextState, nextAction) for nextAction in legalActions])) - self.getQValue(state, action)
    self.weights[feat] = self.weights[feat] + self.alpha * diff * features[feat]
    i += 1
    util.raiseNotDefined()

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      "print(self.weights)"
      pass
