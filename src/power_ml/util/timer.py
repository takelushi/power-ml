"""Timer."""

import logging
import time

from power_ml.util import fmt


class Timer:
    """Logging time."""

    default_message = 'rap'
    default_log_level = logging.INFO
    fmt = 'Rap {n_raps:d}: {rap:8},{total:8} >> {message}'

    def __init__(self, logger: logging.Logger = None) -> None:
        """Initialize object."""
        if logger is None:
            logger = logging.getLogger('timer')
        self.logger = logger
        self.start_time = time.monotonic()
        self.last_time = self.start_time
        self.n_raps = 0

    def rap(self, msg: str = None, log_level=None) -> None:
        """Rap time.

        Args:
            msg (str, optional): Message
            log_level ([type], optional): Log level.
        """
        if msg is None:
            msg = self.default_message

        now = time.monotonic()
        total_timespan = now - self.start_time
        rap_timespan = now - self.last_time

        total_str = fmt.format_timespan(total_timespan)
        rap_str = fmt.format_timespan(rap_timespan)

        if log_level is None:
            log_level = self.default_log_level

        self.n_raps += 1

        text = self.fmt.format(n_raps=self.n_raps,
                               rap=rap_str,
                               total=total_str,
                               message=msg)

        self.logger.log(log_level, text)

        self.last_time = now
