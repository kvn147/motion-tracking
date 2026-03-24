from collections import deque

class MovingAverage:
    def __init__(self, window=5):
        self.values = deque(maxlen=window)

    def update(self, val):
        self.values.append(val)
        return sum(self.values) / len(self.values)