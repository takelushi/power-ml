"""Example of seed."""

import random

from power_ml.util.seed import set_seed, set_seed_random

for v in [5, 5, 6]:
    set_seed(v)
    print('Seed: {}'.format(v))
    print(random.random())
    print(random.random())

v = set_seed_random()
print('Seed: {}'.format(v))
print(random.random())
