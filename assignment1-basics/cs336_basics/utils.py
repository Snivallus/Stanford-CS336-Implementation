# Helper class to tee stdout to multiple streams
class TeeStdout:
    # Initialize with multiple streams
    def __init__(self, *streams):
        self.streams = streams

    # Write data to all streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    # Flush all streams
    def flush(self):
        for s in self.streams:
            s.flush()