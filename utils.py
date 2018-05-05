from timeit import default_timer


class Timer:
    def __enter__(self):
        self.t0 = default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = default_timer()

    @property
    def elapsed_seconds(self):
        return self.t1 - self.t0
