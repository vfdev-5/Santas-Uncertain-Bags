
import sys
sys.path.append('../common')
from utils import validation, weight

import numpy as np


# http://incompleteideas.net/sutton/book/bookdraft2016sep.pdf
#
#
# Approximative Q-learning test
#
#
# State (s)       : gifts in 1000 bags
# Terminal states : bags where we can not add gifts anymore
# Action (a)      : put a gift in a bag
# Reward (r)      : weight of the added gift
# Goal            : Validation score

# Episode : fill bags with gifts
# Policy : how to choose gifts
#
# Q(s,a) : action-value function
#
# page 158 (140), Q-learning: Off-policy TD Control
# Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * [ r + gamma * max_{a'} Q(s',a')]
#
#
# Q(s,a) = w0 * (total weight of all bags) + w1 * (added gift weight from action)
#

alpha = 0.10  # ...
gamma = 0.9  # discount rate

# State parameters
# n_bags = 1000
n_bags = 50
max_weight = 50

n_horses = 1000
n_balls = 1100
n_bikes = 500
n_trains = 1000
n_coals = 166
n_books = 1200
n_dolls = 1000
n_blocks = 1000
n_gloves = 200

available_gifts = {
    "horse": n_horses,
    "ball": n_balls,
    "bike": n_bikes,
    "train": n_trains,
    "coal": n_coals,
    "book": n_books,
    "doll": n_dolls,
    "blocks": n_blocks,
    "gloves": n_gloves
}
gift_types = available_gifts.keys()
# n_types = 8
n_types = 3

W = [0.0, 0.0]
n_factor = max_weight * n_bags


def get_feature(i, state, action):
    if i == 0:
        f0 = 0
        for bag_id in state:
            total_weight, gifts = state[bag_id]
            f0 += total_weight
        return f0 * 1.0 / n_factor
    elif i == 1:
        if action is not None:
            gift_type, bag_id = action
            return weight(gift_type) * 1.0 / n_factor
        else:
            return 0.0

    raise Exception("Feature is not defined")


def get_q(state, action):
    """
    Method computes Q(s,a).
    :param state: bags
    :param action:
    :return:
    """
    q = 0
    for i in range(2):
        q += W[i] * get_feature(i, state, action)

    return q


def get_action(state):
    """
    Method to get an action from a state using a policy
    :param state:
    :return: we pick a random gift type and random bag index where to add the gift
    """
    counter = 0
    while counter < n_types**3:
        counter += 1
        gift_index = np.random.randint(0, n_types)
        if gift_types[gift_index] > 0:
            return gift_types[gift_index], np.random.randint(0, n_bags)

    raise Exception("Failed to get an action")


def apply_action(action, state):
    """
    Method to take the action from state
    :param action:
    :param state:
    :return: new state, reward
    """
    reward = 0
    gift_type, bag_id = action
    gift_weight = weight(gift_type)
    new_state = state.copy()
    if bag_id not in state:
        gift = gift_type + '_%i' % available_gifts[gift_type]
        available_gifts[gift_type] -= 1
        new_state[bag_id] = [gift_weight, [gift]]
        reward = gift_weight * 0.1
    else:
        total_weight, gifts = state[bag_id]
        if total_weight + gift_weight < max_weight:
            new_state[bag_id][0] += gift_weight
            gift = gift_type + '_%i' % available_gifts[gift_type]
            new_state[bag_id][1].append(gift)
            reward = gift_weight * 0.1

    return new_state, reward


def is_terminal(state):
    for bag_id in state:
        total_weight, _ = state[bag_id]
        if total_weight < max_weight * 0.95:
            return False
    return True


def compute_max_q(state):
    max_q = 0
    counter = n_bags/10
    while counter > 0:
        counter -= 1
        action = get_action(state)
        max_q = max(get_q(state, action), max_q)

    return max_q

# Run learning algorithm
n_episodes = 1000


for i in range(n_episodes):
    print "\n\n-------------------"
    print "--- Episode : ", i, "---"
    print "-------------------\n\n"

    # initialize state :
    bags = {}
    is_running = True
    counter = 0
    while is_running:
        counter += 1
        action = get_action(bags)
        # print "- action : ", action
        new_bags, reward = apply_action(action, bags)
        # print "-- new state : ", new_bags
        # print "-- reward : ", reward

        if is_terminal(new_bags):
            is_running = False
            # print "new_bags : ", new_bags
            goal_reward = validation(new_bags, weight) * 1.0 / n_bags
            if goal_reward.mean() < max_weight * 0.75:
                print "-- Bad terminal state"
                reward -= 100
            else:
                print "-- Good terminal state"
                reward += 100
            max_q = 0
        else:
            max_q = compute_max_q(new_bags)

        prev_q = get_q(bags, action)
        # print "- Q(state, action) : ", prev_q
        delta = (reward + gamma * max_q - prev_q)
        # print "-- delta, reward, max_q : ", delta, reward, max_q
        for i in range(len(W)):
            W[i] += alpha * delta * get_feature(i, bags, action)

        if (counter % 500) == 0:
            print "- New Q(state, action) : ", get_q(bags, action)

        bags = new_bags
        # if counter == 10:
        #     is_running = False

    print " End of episode : ", validation(bags, weight)






