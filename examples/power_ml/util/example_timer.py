"""Example of timer."""

import logging
import time

from power_ml.util.timer import Timer

logging.basicConfig(level=logging.DEBUG)
timer = Timer()

for msg in ['One', 'Two', 'Three']:
    time.sleep(0.2)
    timer.rap(msg)

time.sleep(0.1)
timer.rap()  # Rap without message.

# Format.
timer = Timer()
timer.fmt = '[{n_raps:d}] {rap} - {total} | {message}'
timer.default_message = 'RAP!'
timer.default_log_level = logging.DEBUG

for msg in ['One', 'Two', 'Three']:
    time.sleep(0.2)
    timer.rap(msg)
time.sleep(0.1)
timer.rap()
