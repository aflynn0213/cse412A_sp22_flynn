# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    root = problem.getStartState()
    node = root
    print "Is the start a goal?"

    if problem.isGoalState(root):
        return True
    else:
        print "False"

    dfs_tree = util.Stack()
    _visited = []
    node_root = (root,[])
    dfs_tree.push(node_root)

    while (not dfs_tree.isEmpty() and not problem.isGoalState(node)):
        curr_node, directions = dfs_tree.pop()
        children = problem.getSuccessors(curr_node)
        _visited.append(curr_node)

        for child in children:

            if (not child[0] in _visited):
                node = child[0]
                curr_dir = directions+[child[1]]
                dfs_tree.push((node,curr_dir))

    return curr_dir

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    root = problem.getStartState()
    bfs_tree = util.Queue()
    _visited = []
    bfs_tree.push((root,[]))

    while not bfs_tree.isEmpty():
        curr_node, directions = bfs_tree.pop()

        if curr_node in _visited:
            continue

        if problem.isGoalState(curr_node):
            return directions

        children = problem.getSuccessors(curr_node)
        _visited.append(curr_node)

        for child in children:
            bfs_tree.push((child[0],directions+[child[1]]))

    util.raiseNotDefined()

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()
    heap = util.PriorityQueue()
    heap.push((root,[]),0)
    _visited = []

    while not heap.isEmpty():
        curr_node, directions = heap.pop()

        if curr_node in _visited:
            continue

        if problem.isGoalState(curr_node):
            return directions

        children = problem.getSuccessors(curr_node)
        _visited.append(curr_node)

        for child in children:
            curr_dir = directions + [child[1]]
            heap.push((child[0],curr_dir),problem.getCostOfActions(curr_dir))

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()
    heap = util.PriorityQueue()
    heap.push((root,[]),0)
    _visited = []

    while not heap.isEmpty():
        curr_node, directions = heap.pop()

        if (curr_node in _visited):
            continue

        if problem.isGoalState(curr_node):
            return directions

        children = problem.getSuccessors(curr_node)
        _visited.append(curr_node)

        for child in children:
            curr_dir = directions + [child[1]]
            heap.push((child[0],curr_dir),problem.getCostOfActions(curr_dir)+heuristic(child[0],problem))

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
