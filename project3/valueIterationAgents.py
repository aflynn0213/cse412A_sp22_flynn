# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 6):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    
    for i in range(iterations):
      temp_vals = util.Counter()
      for s in mdp.getStates():
        actions = mdp.getPossibleActions(s)
        temp_vals = self.calcTransProbsQVals(s,actions)
        self.values[s] = temp_vals[temp_vals.argMax()]
        
        
  def calcTransProbsQVals(self,state,actions):
    temp_vals = util.Counter()
    for act in actions:
          t_s = self.mdp.getTransitionStatesAndProbs(state,act)
          temp_val = 0
          for t in t_s:
            temp_val += self.internalQValCalc(state,act,t)
          temp_vals[act] = temp_val
    return temp_vals

  def internalQValCalc(self,s,act,t):
    return t[1]*(self.mdp.getReward(s,act,t[0]) + self.discount*self.values[t[0]])

  
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action, qVal=0):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    states_probs = self.mdp.getTransitionStatesAndProbs(state, action)
    
    for t in states_probs:
      qVal += t[1] * self.mdp.getReward(state, action, t[0]) + self.values[t[0]]

    return qVal
    util.raiseNotDefined()

 

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    if self.mdp.isTerminal(state):
      return None

    acts = self.mdp.getPossibleActions(state)
    counter = util.Counter()
    for i in acts:
      trans = self.mdp.getTransitionStatesAndProbs(state,i)
      sum = 0
      for t in trans:
        sum += self.internalQValCalc(state,i,t)
      counter[i] = sum

    return counter.argMax()
  
    util.raiseNotDefined()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
