
import numpy as np

MAX_WEIGHT = 50
N_BAGS = 1000

N_HORSES = 1000
N_BALLS = 1100
N_BIKES = 500
N_TRAINS = 1000
N_COALS = 166
N_BOOKS = 1200
N_DOLLS = 1000
N_BLOCKS = 1000
N_GLOVES = 200

AVAILABLE_GIFTS = {
    "horse": N_HORSES,
    "ball": N_BALLS,
    "bike": N_BIKES,
    "train": N_TRAINS,
    "coal": N_COALS,
    "book": N_BOOKS,
    "doll": N_DOLLS,
    "blocks": N_BLOCKS,
    "gloves": N_GLOVES
}

GIFT_TYPES = sorted(list(AVAILABLE_GIFTS.keys()))
N_TYPES = len(GIFT_TYPES)


def validation(bags, weigth_func, count=5):
    """
    Method to compute total weights of the bags using weight_func count times
    :param bags:
    :param weigth_func:
    :param count:
    :return: a vector of total weights
    """
    scores = np.zeros(count)
    for c in range(count):
        score = 0
        for bag_id in bags:
            _, gifts = bags[bag_id]
            total_weight_ = 0
            for g in gifts:
                gift_type = g.split('_')[0]
                total_weight_ += weigth_func(gift_type)

            if total_weight_ < 50.0:
                score += total_weight_

        scores[c] = score
    return scores


def weight(gift_type):
    if gift_type == "horse":
        return max(0, np.random.normal(5, 2, 1)[0])
    if gift_type == "ball":
        return max(0, 1 + np.random.normal(1, 0.3, 1)[0])
    if gift_type == "bike":
        return max(0, np.random.normal(20, 10, 1)[0])
    if gift_type == "train":
        return max(0, np.random.normal(10, 5, 1)[0])
    if gift_type == "coal":
        return 47 * np.random.beta(0.5, 0.5, 1)[0]
    if gift_type == "book":
        return np.random.chisquare(2,1)[0]
    if gift_type == "doll":
        return np.random.gamma(5, 1, 1)[0]
    if gift_type == "blocks":
        return np.random.triangular(5, 10, 20, 1)[0]
    if gift_type == "gloves":
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]


def weight2(gift_type, count=1000):
    w = []
    for c in range(count):
        w.append(weight(gift_type))
    return np.mean(w)


def weight_by_index(index):
    """
    index :   0     1      2    3     4     5       6      7      8
            ball, bike, block, book, coal, doll, gloves, horse, train
    :param index:
    :return:
    """
    if index == 0: return max(0, 1 + np.random.normal(1, 0.3, 1)[0])
    elif index == 1: return max(0, np.random.normal(20, 10, 1)[0])
    elif index == 2: return np.random.triangular(5, 10, 20, 1)[0]
    elif index == 3: return np.random.chisquare(2,1)[0]
    elif index == 4: return 47 * np.random.beta(0.5, 0.5, 1)[0]
    elif index == 5: return np.random.gamma(5, 1, 1)[0]
    elif index == 6: return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
    elif index == 7: return max(0, np.random.normal(5, 2, 1)[0])
    elif index == 8: return max(0, np.random.normal(10, 5, 1)[0])
    raise Exception("Index is out of bounds")


def weight3(index, count=100):
    w = []
    for c in range(count):
        w.append(weight_by_index(index))
    return np.mean(w)


def bag_weight(bag, n1=100):
    weight = 0
    for index, count in enumerate(bag):
        for i in range(count):
            weight += weight3(index, n1)
    return weight


def score(state, count=100, max_weight=MAX_WEIGHT, return_rejected=False):
    scores = np.zeros(count)
    rejected_bags = np.zeros(count)
    for c in range(count):
        score = 0
        rejected = 0
        for bag in state:
            total_weight_ = bag_weight(bag, n1=1)
            if total_weight_ < max_weight:
                score += total_weight_
            else:
                rejected += 1
        rejected_bags[c] = rejected
        scores[c] = score
    if return_rejected:
        return np.mean(scores), np.mean(rejected_bags)
    else:
        return np.mean(scores)

