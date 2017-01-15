
import numpy as np
import sys

# Project
sys.path.append('../common')
from utils import MAX_WEIGHT, AVAILABLE_GIFTS_LIST, GIFT_TYPES, N_TYPES, N_BAGS
from search import Problem


class SantasBagsProblem2(Problem):

    def actions(self, state):
        """Return a list of actions executable in this state."""
        # all_actions = []
        # for index, t in enumerate(state):
        #     if t < self.available_gifts[index]:
        #         all_actions.append(index)
        # if self.verbose_level >= 2:
        #     print("-- actions : ", all_actions)

        # actions = []
        # for index in all_actions:
        #     probe_state = list(state)
        #     probe_state[index] += 1
        #     weight = self.bag_weight_fn(probe_state)
        #     if weight < self.max_weight:
        #         actions.append(index)

        # actions = [i for i in range(len(state))]
        actions = []
        for index in range(len(state)):
            probe_state = list(state)
            probe_state[index] += 1
            weight = self.bag_weight_fn(probe_state)
            if weight < self.max_weight:
                actions.append(index)


        return actions

    def result(self, state, action):
        """The state that results from executing this action in this state."""
        if self.verbose_level >= 2:
            print("-- result : input state: ", state, "action: ", action)
        new_state = list(state)
        new_state[action] += 1
        if self.verbose_level >= 2:
            print("-- result : output state: ", new_state)
        return tuple(new_state)

    def is_goal(self, state):
        """True if the state is a goal."""
        if sum(state) < 3:
            # print("- A bag with less than 3 gifts found : ", state)
            return False

        # Check if solution is available:
        for index, t in enumerate(state):
            if t > self.available_gifts[index]:
                return False

        mean_score = self.score_fn(state)
        if self.verbose_level >= 1:
            print("- Mean score : ", mean_score, " / ", self.goal_score, state)
        return mean_score > self.goal_score

    def step_cost(self, state, action, result=None):
        """The cost of taking this action from this state."""
        return 0.0


def update_available_gifts(available_gifts, state):
    for i, t in enumerate(state):
        assert available_gifts[i] - t >= 0, "Found state is not available : {}, {}".format(state, available_gifts)
        available_gifts[i] = available_gifts[i] - t


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
            print(">>> Current score: ", s, s * (n_bags - 100) / (n_bags_per_state * counter[0]))
            last_score_computation = counter[0]

        if counter[0] > 0 and (n_bags_per_state * counter[0] % 30) == 0 and \
                last_score_computation < counter[0]:
            print(">>> Currently available gifts : ", [(k, available_gifts[k]) for k in gift_types])
            last_score_computation = counter[0]

        # print("- Elapsed: ", time() - tic)
