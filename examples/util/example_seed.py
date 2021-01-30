"""Example of seed."""

import random

from power_ml.util import seed

for v in [5, 5, 6]:
    seed.set_seed(v)
    print('Seed: {}'.format(v))
    print(random.random())
    print(random.random())

v = seed.set_seed_random()
print('Seed: {}'.format(v))
print(random.random())
