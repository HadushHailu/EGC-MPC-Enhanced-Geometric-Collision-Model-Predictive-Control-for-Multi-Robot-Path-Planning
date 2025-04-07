import time

class Rate:
    def __init__(self, frequency_hz):
        self.period = 1.0 / frequency_hz
        self.next_time = time.time() + self.period

    def sleep(self):
        """Sleep just long enough to maintain the loop rate."""
        now = time.time()
        sleep_time = self.next_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.next_time += self.period
