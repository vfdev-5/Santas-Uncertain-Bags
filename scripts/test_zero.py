
import sys
from datetime import datetime
import numpy as np

sys.path.append('../common')
from utils import validation

np.random.seed(2016)

# Lbs to Kg
lbs_kg = 0.454
kg_lbs = 1.0/lbs_kg


max_weights = {"horse": 10.0 * kg_lbs, "ball": 1.0 * kg_lbs, "bike": 15.0 * kg_lbs, "train": 3.0 * kg_lbs,
              "coal": 50.0, "book": 2.0 * kg_lbs, "doll": 2.0 * kg_lbs, "blocks": 50.0, "gloves": 0.3 * kg_lbs}

min_weights = {"horse": 0.1 * kg_lbs, "ball": 0.1 * kg_lbs, "bike": 7.0 * kg_lbs, "train": 0.1 * kg_lbs, "coal": 0.001,
              "book": 0.1 * kg_lbs, "doll": 0.1 * kg_lbs, "blocks": 0.001, "gloves": 0.05 * kg_lbs }


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


data_file = '../input/gifts.csv'
submission_file = '../results/submission_' + \
                  str(datetime.now().strftime("%Y-%m-%d-%H-%M")) + \
                  '.csv'


#  bags = {bag_id: (total_weight,[gift_1, gift_2, ...]), ... }
bags = {}
n_bags = 1000
bag_id = 0
max_weight = 50

counter = 0
with open(data_file, 'r') as r:

    header = r.readline()
    print header

    while True:

        gift = r.readline().strip()

        if gift == '':
            break
        gift_type = gift.split('_')[0]
        gift_weight = weight(gift_type)

        if bag_id not in bags:
            bags[bag_id] = [gift_weight, [gift]]
        else:
            total_weight, gifts = bags[bag_id]
            if total_weight + gift_weight < max_weight:
                bags[bag_id][0] += gift_weight
                bags[bag_id][1].append(gift)
            else:
                counter += 1

        bag_id = bag_id+1 if bag_id+1 < n_bags else 0

print 'Number of not added gifts: ', counter

# Validation :
scores = validation(bags, weight, 5)
print "Score: ", scores.mean()



# with open(submission_file, 'w') as w:
#     w.write("Gifts\n")
#     for bag_id in bags:
#         _, gifts = bags[bag_id]
#         w.write(' '.join(gifts) + '\n')


##  Submission: submission_2016-12-23-19-02.csv
##       Score: 41243.9577061
##      Kaggle: 27047.92870
