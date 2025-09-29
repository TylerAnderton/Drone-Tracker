import time
from contextlib import contextmanager


def now_ms() -> float:
    return time.perf_counter() * 1000.0


@contextmanager
def timer_ctx(name: str = ""):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        if name:
            print(f"[{name}] {dt:.2f} ms")


class Timer:
    def __init__(self):
        self.t0 = None
        self.dt = 0.0

    def tic(self):
        self.t0 = time.perf_counter()

    def toc(self) -> float:
        if self.t0 is None:
            return 0.0
        self.dt = (time.perf_counter() - self.t0) * 1000.0
        return self.dt
