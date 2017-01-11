
from datetime import datetime
import numpy as np

import sys
sys.path.append('../common')
from utils import validation
from utils import MAX_WEIGHT, N_BAGS

np.random.seed(2016)

data_file = '../input/gifts.csv'
submission_file = '../results/submission_' + \
                  str(datetime.now().strftime("%Y-%m-%d-%H-%M")) + \
                  '.csv'


fix_weight = {
    'ball': 1.70,
    'bike': 20.86,
    'blocks': 12.0,
    'book': 1.9,
    'coal': 45,
    'doll': 4.9,
    'gloves': 1.21,
    'horse': 4.83,
    'train': 9.6
}

#  bags = {bag_id: (total_weight,[gift_1, gift_2, ...]), ... }
bags = {}
bag_id = 0
alpha = 0.7
goal_weight = alpha * MAX_WEIGHT

total_counter = 0
counter = 0
not_added_gifts = []
with open(data_file, 'r') as r:
    header = r.readline()
    print header

    while True:
        gift = r.readline().strip()
        if gift == '':
            break
        gift_type = gift.split('_')[0]
        gift_weight = fix_weight[gift_type]

        if bag_id not in bags:
            bags[bag_id] = [gift_weight, [gift]]
        else:
            total_weight, gifts = bags[bag_id]
            if total_weight + gift_weight < goal_weight:
                bags[bag_id][0] += gift_weight
                bags[bag_id][1].append(gift)
            else:
                not_added_gifts.append(gift)
                counter += 1

        bag_id = bag_id+1 if bag_id+1 < N_BAGS else 0
        total_counter += 1

print 'Number of not added gifts: {}/{}'.format(counter, total_counter)

counter = 0
for gift in not_added_gifts:
    gift_type = gift.split('_')[0]
    gift_weight = fix_weight[gift_type]
    for bag_id in bags:
        total_weight, gifts = bags[bag_id]
        if total_weight + gift_weight < goal_weight:
            bags[bag_id][0] += gift_weight
            bags[bag_id][1].append(gift)
            counter += 1
            break


print 'Number of not added gifts: {}/{}'.format(len(not_added_gifts) - counter, total_counter)


# Validation :
scores = validation(bags, count=5)
print "Score: ", scores.mean()

print "Filled bags : ", len(bags)

# with open(submission_file, 'w') as w:
#     w.write("Gifts\n")
#     for bag_id in bags:
#         _, gifts = bags[bag_id]
#         w.write(' '.join(gifts) + '\n')


##  Submission:
##       Score: 31716
##      Kaggle:


