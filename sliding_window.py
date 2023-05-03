
class SlidingWindowEncoder:
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size

    def encode(self, input_data):
        if len(input_data) <= self.window_size:
            return [input_data]
        windows = []
        start = 0
        end = self.window_size
        while end <= len(input_data):
            windows.append(input_data[start:end])
            start += self.step_size
            end += self.step_size
        if start < len(input_data):
            windows.append(input_data[start:])
        return windows

    def decode(self, encoded_data):
        decoded_data = []
        for window in encoded_data:
            decoded_data.extend(window)
        return decoded_data