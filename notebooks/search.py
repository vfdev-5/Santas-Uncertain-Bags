#!/usr/bin/python3
import heapq
import random
import math
import sys


class Node(object):
    """A node in a search tree. A search tree is spanning tree over states.
    A Node contains a state, the previous node in the tree, the action that
    takes us from the previous state to this state, and the path cost to get to
    this state. If a state is arrived at by two paths, then there are two nodes
    with the same state."""

    def __init__(self, state, previous=None, action=None, step_cost=1):
        """Create a search tree Node, derived from a previous Node by an action."""
        self.state = state
        self.previous = previous
        self.action = action
        self.path_cost = 0 if previous is None else (previous.path_cost + step_cost)

    def __repr__(self): return "<Node {}: {}>".format(self.state, self.path_cost)

    def __lt__(self, other): return self.path_cost < other.path_cost

    def child(self, problem, action):
        """The Node you get by taking an action from this Node."""
        result = problem.result(self.state, action)
        return Node(result, self, action,
                    problem.step_cost(self.state, action, result))

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child(problem, action) for action in problem.actions(self.state)]


class FrontierPQ:
    """A Frontier ordered by a cost function; a Priority Queue."""

    def __init__(self, initial, costfn=lambda node: node.path_cost):
        """Initialize Frontier with an initial Node, and specify a cost function."""
        self.heap = []
        self.states = {}
        self.costfn = costfn
        self.add(initial)

    def add(self, node):
        """Add node to the frontier."""
        cost = self.costfn(node)
        heapq.heappush(self.heap, (cost, node))
        self.states[node.state] = node

    def pop(self):
        """Remove and return the Node with minimum cost."""
        (cost, node) = heapq.heappop(self.heap)
        self.states.pop(node.state, None)  # remove state
        return node

    def replace(self, node):
        """Make this node replace a previous node with the same state."""
        if node.state not in self:
            raise ValueError('{} not there to replace'.format(node.state))
        for (i, (cost, old_node)) in enumerate(self.heap):
            if old_node.state == node.state:
                self.heap[i] = (self.costfn(node), node)
                heapq._siftdown(self.heap, 0, i)
                return

    def __contains__(self, state):
        return state in self.states

    def __len__(self):
        return len(self.heap)


class Problem(object):
    """The abstract class for a search problem."""

    def __init__(self, initial=None, goals=(), **additional_keywords):
        """Provide an initial state and optional goal states.
        A subclass can have additional keyword arguments."""
        self.initial = initial  # The initial state of the problem.
        self.goals = goals      # A collection of possibe goal states.
        self.__dict__.update(**additional_keywords)

    def actions(self, state):
        """Return a list of actions executable in this state."""
        raise NotImplementedError  # Override this!

    def result(self, state, action):
        """The state that results from executing this action in this state."""
        raise NotImplementedError  # Override this!

    def is_goal(self, state):
        """True if the state is a goal."""
        return state in self.goals  # Optionally override this!

    def step_cost(self, state, action, result=None):
        """The cost of taking this action from this state."""
        return 1  # Override this if actions have different costs

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


def action_sequence(node):
    """The sequence of actions to get to this node."""
    actions = []
    while node.previous:
        actions.append(node.action)
        node = node.previous
    return actions[::-1]


def state_sequence(node):
    """The sequence of states to get to this node."""
    states = [node.state]
    while node.previous:
        node = node.previous
        states.append(node.state)
    return states[::-1]


def uniform_cost_search(problem, costfn=lambda node: node.path_cost):
    frontier = FrontierPQ(Node(problem.initial), costfn)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        explored.add(node.state)
        for action in problem.actions(node.state):
            child = node.child(problem, action)
            if child.state not in explored and child not in frontier:
                frontier.add(child)
            elif child in frontier and frontier.cost[child] < child.path_cost:
                frontier.replace(child)


def astar_search(problem, heuristic):
    costfn = lambda node: node.path_cost + heuristic(node.state)
    return uniform_cost_search(problem, costfn)


argmin = min
argmax = max
identity = lambda x: x


def shuffled(iterable):
    "Randomly shuffle a copy of iterable."
    items = list(iterable)
    random.shuffle(items)
    return items


def argmax_random_tie(seq, key=identity):
    "Return an element with highest fn(seq[i]) score; break ties at random."
    return argmax(shuffled(seq), key=key)


def hill_climbing(problem):
    """From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better. [Figure 4.2]"""
    current = Node(problem.initial)
    while True:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmax_random_tie(neighbors,
                                     key=lambda node: problem.value(node.state))
        if problem.value(neighbor.state) <= problem.value(current.state):
            break
        current = neighbor
    return current.state


def exp_schedule(k=20, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * math.exp(-lam * t) if t < limit else 0)


def probability(p):
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)


def simulated_annealing(problem, schedule=exp_schedule()):
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current
        neighbors = current.expand(problem)
        if not neighbors:
            return current
        next = random.choice(neighbors)
        delta_e = problem.value(next.state) - problem.value(current.state)
        if delta_e > 0 or probability(math.exp(delta_e / T)):
            current = next