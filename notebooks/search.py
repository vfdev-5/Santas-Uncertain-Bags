#!/usr/bin/python3
import heapq
import random
import math
import sys

import numpy as np
from time import time

# Project
sys.path.append('../common')
from utils import MAX_WEIGHT, AVAILABLE_GIFTS, GIFT_TYPES, N_TYPES, N_BAGS


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


def uniform_cost_search(problem, costfn=lambda node: node.path_cost, **kwargs):
    verbose = True if 'verbose' in kwargs and kwargs['verbose'] else False
    frontier = FrontierPQ(Node(problem.initial), costfn)
    explored = set()
    while frontier:
        node = frontier.pop()
        if verbose:
            print("Check node: ", node.state, " | ", node.path_cost, ", ", costfn(node))
        if problem.is_goal(node.state):
            return node
        explored.add(node.state)
        for action in problem.actions(node.state):
            child = node.child(problem, action)
            if child.state not in explored and child.state not in frontier:
                frontier.add(child)
            elif child.state in frontier:
                incumbent = frontier.states[child.state]
                if child.path_cost < frontier.costfn(incumbent):
                    frontier.replace(child)


def astar_search(problem, heuristic, **kwargs):
    costfn = lambda node: node.path_cost + heuristic(node.state)
    return uniform_cost_search(problem, costfn, **kwargs)


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


class SantasBagsProblem(Problem):
    def _get_gift_type_indices(self, state):
        out = []
        types = np.sum(np.array(state), axis=0)
        for index, t in enumerate(types):
            if t < self.available_gifts[self.gift_types[index]]:
                out.append(index)
        return out

    def actions(self, state):
        """Return a list of actions executable in this state."""
        _gift_type_indices = self._get_gift_type_indices(state)
        if len(_gift_type_indices) == 0:
            print("No gifts available to create actions")
            return []

        if self.verbose_level >= 2:
            print("_gift_type_indices : ", _gift_type_indices)
        # find a bag with a minimal weight
        min_weight_bag_index = 0
        min_weight = self.max_weight
        for i, bag in enumerate(state):
            w = self.bag_weight_fn(bag)
            if min_weight > w:
                min_weight_bag_index = i
                min_weight = w

        if self.verbose_level >= 2:
            print("min_weight_bag_index : ", min_weight_bag_index)

        actions = []
        bag_weight = self.bag_weight_fn(state[min_weight_bag_index])
        for _index in _gift_type_indices:
            gift_weight = self.gift_weight_fn(_index)
            if bag_weight + gift_weight < self.max_weight:
                actions.append((min_weight_bag_index, _index))

        if self.verbose_level >= 2:
            print("actions: ", actions)
            if len(actions) == 0:
                print("No actions found for the state : ", state, min_weight_bag_index, bag_weight)

        return actions

    def result(self, state, action):
        """The state that results from executing this action in this state."""
        bag_id, gift_type_index = action
        if self.verbose_level >= 2:
            print("-- result : input state: ", state, "action: ", action)
        new_state = list(state)
        bag = list(new_state[bag_id])
        bag[gift_type_index] += 1
        new_state[bag_id] = tuple(bag)
        if self.verbose_level >= 2:
            print("-- result : output state: ", new_state)
        return tuple(new_state)

    def is_goal(self, state):
        """True if the state is a goal."""
        for bag in state:
            if sum(bag) < 3:
                # print("- A bag with less than 3 gifts found : ", state)
                return False

        # Check if solution is available:
        types = np.sum(np.array(state), axis=0)
        for index, t in enumerate(types):
            if t > self.available_gifts[self.gift_types[index]]:
                return False

        mean_score = self.score_fn(state)
        if self.verbose_level >= 1:
            print("- Mean score : ", mean_score, " / ", self.goal_score, state)
        return mean_score > self.goal_score

    def step_cost(self, state, action, result=None):
        """The cost of taking this action from this state."""
        if self.type_cost is not None:
            bag_id, gift_type_index = action
            gift_type = self.gift_types[gift_type_index]
            if gift_type in self.type_cost:
                return self.type_cost[gift_type]  # Override this if actions have different costs
            return 1.0
        return 1.0


def update_available_gifts(available_gifts, state, gift_types=GIFT_TYPES):
    sum_gifts = np.sum(np.array(state), axis=0)
    for v, gift_type in zip(sum_gifts, gift_types):
        assert available_gifts[gift_type] - v >= 0, "Found state is not available : {}, {}".format(state, ag)
        available_gifts[gift_type] = available_gifts[gift_type] - v


def remove_gifts(state, empty_state, gifts_to_remove=2, n_types=N_TYPES,  **kwargs):
    _gift_removed = 0
    new_state = list(state)
    for bag_index, bag in enumerate(state):
        for i in range(gifts_to_remove):
            gift_type_index = np.argmax(bag)
            if bag[gift_type_index] > 0:
                bag = list(new_state[bag_index])
                bag[gift_type_index] -= 1
                new_state[bag_index] = tuple(bag)
                _gift_removed += 1
    if _gift_removed == 0:
        state = empty_state
    else:
        print("-- Remove some gift : ", state, tuple(new_state))
        state = tuple(new_state)
    return state


def on_result_none(p):
    p.goal_score -= 0.05


def update_problem(p, state, available_gifts,
                    counter, total_state,
                    found_goal_states,
                    **kwargs):
    p.initial = state
    p.available_gifts = available_gifts


def termination_condition(p, counter, total_state):
    if p.goal_score < 18:
        return True
    return False


def fill_all_bags(create_problem_fn, search_algorithm_fn,
                    total_state, found_goal_states, available_gifts, counter,
                    score_fn=None,
                    gift_types=GIFT_TYPES,
                    n_bags_per_state=1,
                    n_types=N_TYPES,
                    n_bags=N_BAGS,
                    **kwargs):

    empty_state = tuple([tuple([0] * n_types)] * n_bags_per_state)
    state = (total_state[-1],) if len(total_state) > 0 else empty_state

    last_score_computation = -1

    p = create_problem_fn(state, available_gifts, **kwargs)
    update_problem_fn = update_problem if 'update_problem_fn' not in kwargs else kwargs['update_problem_fn']
    termination_condition_fn = termination_condition if 'termination_condition_fn' not in kwargs \
        else kwargs['termination_condition_fn']
    on_result_none_fn = on_result_none if 'on_result_none_fn' not in kwargs else kwargs['on_result_none_fn']

    while n_bags_per_state * counter[0] < n_bags and \
            not termination_condition_fn(p, counter, total_state):
        print("Filled bags : ", n_bags_per_state * counter[0], "/", n_bags)

        update_problem_fn(p, state, available_gifts, counter, total_state,
                          found_goal_states, **kwargs)

        # tic = time()
        result = search_algorithm_fn(p, **kwargs)
        if result is not None:
            print("- Got a result : ", p.goal_score)
            update_available_gifts(available_gifts, result.state, gift_types)
            if len(found_goal_states) == 0 or found_goal_states[-1] != result.state:
                found_goal_states.append(result.state)
            total_state += result.state
            counter[0] += 1
            state = (total_state[-1],)
        else:
            print("-- Result is none | len(found_goal_states)=", len(found_goal_states))
            if len(found_goal_states) > 0:
                state = found_goal_states.pop()
                print("--- Restart from : ", state)
            else:
                if state != empty_state:
                    state = remove_gifts(state, empty_state, **kwargs)
                else:
                    on_result_none_fn(p)
                    print(">>> Goal score changed: ", p.goal_score)

        if counter[0] > 0 and (n_bags_per_state * counter[0] % 20) == 0 and \
            score_fn is not None and last_score_computation < counter[0]:
            s = score_fn(total_state)
            print(">>> Current score: ", s, s * n_bags / (n_bags_per_state * counter[0]))
            last_score_computation = counter[0]

        if counter[0] > 0 and (n_bags_per_state * counter[0] % 30) == 0 and \
                last_score_computation < counter[0]:
            print(">>> Currently available gifts : ", [(k, available_gifts[k]) for k in gift_types])
            last_score_computation = counter[0]

        # print("- Elapsed: ", time() - tic)
