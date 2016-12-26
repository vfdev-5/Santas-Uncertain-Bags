
from datetime import datetime
import numpy as np

import sys
sys.path.append('../common')
from utils import validation


np.random.seed(2016)

n_horses = 1000
n_balls = 1100
n_bikes = 500
n_trains = 1000
n_books = 1200
n_dolls = 1000
n_blocks = 1000
n_gloves = 200

# Lbs to Kg
lbs_kg = 0.454
kg_lbs = 1.0/lbs_kg


max_weights = {"horse": 10.0 * kg_lbs, "ball": 1.0 * kg_lbs, "bike": 15.0 * kg_lbs, "train": 3.0 * kg_lbs,
              "coal": 10.0, "book": 2.0 * kg_lbs, "doll": 2.0 * kg_lbs, "blocks": 10.0, "gloves": 0.3 * kg_lbs}

min_weights = {"horse": 0.1 * kg_lbs, "ball": 0.1 * kg_lbs, "bike": 7.0 * kg_lbs, "train": 0.1 * kg_lbs, "coal": 0.001,
              "book": 0.1 * kg_lbs, "doll": 0.1 * kg_lbs, "blocks": 0.001, "gloves": 0.05 * kg_lbs}


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
        return np.random.chisquare(2, 1)[0]
    if gift_type == "doll":
        return np.random.gamma(5, 1, 1)[0]
    if gift_type == "blocks":
        return np.random.triangular(5, 10, 20, 1)[0]
    if gift_type == "gloves":
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]


submission_file = '../results/submission_' + \
                  str(datetime.now().strftime("%Y-%m-%d-%H-%M")) + \
                  '.csv'


#  bags = {bag_id: (total_weight,[gift_1, gift_2, ...]), ... }
bags = {}
n_bags = 1000
bag_id = 0
max_weight = 48

total_counter = 0
counter = 0

not_added_gifts = []

while True:

    gift = r.readline().strip()

    if gift == '':
        break
    gift_type = gift.split('_')[0]
    gift_weight = weight(gift_type)
    # gift_weight = max(min(weight(gift_type), max_weights[gift_type]), min_weights[gift_type])
    # gift_weight = max_weights[gift_type]

    if bag_id not in bags:
        bags[bag_id] = [gift_weight, [gift]]
    else:
        total_weight, gifts = bags[bag_id]
        if total_weight + gift_weight < max_weight:
            bags[bag_id][0] += gift_weight
            bags[bag_id][1].append(gift)
        else:
            not_added_gifts.append(gift)
            counter += 1

    bag_id = bag_id+1 if bag_id+1 < n_bags else 0
    total_counter += 1

print 'Number of not added gifts: {}/{}'.format(counter, total_counter)

counter = 0
for gift in not_added_gifts:
    gift_type = gift.split('_')[0]
    gift_weight = weight(gift_type)
    for bag_id in bags:
        total_weight, gifts = bags[bag_id]
        if total_weight + gift_weight < max_weight:
            bags[bag_id][0] += gift_weight
            bags[bag_id][1].append(gift)
            counter += 1
            break


print 'Number of not added gifts: {}/{}'.format(len(not_added_gifts) - counter, total_counter)


# Validation :
scores = validation(bags, weight, 5)
print "Score: ", scores.mean()

# with open(submission_file, 'w') as w:
#     w.write("Gifts\n")
#     for bag_id in bags:
#         _, gifts = bags[bag_id]
#         w.write(' '.join(gifts) + '\n')


##  Submission:
##       Score:
##      Kaggle:


