import os
from contextlib import contextmanager, ExitStack
from typing import Generator

from torch.distributed.elastic.multiprocessing.errors import record

TORCH_WORKER_SERVER_SOCKET = "TORCH_WORKER_SERVER_SOCKET"


@contextmanager
def _worker_server(socket_path: str) -> Generator[None, None, None]:
    from torch._C._distributed_c10d import _WorkerServer

    server = _WorkerServer(socket_path)
    try:
        yield
    finally:
        server.shutdown()


@contextmanager
@record
def main() -> Generator[None, None, None]:
    with ExitStack() as stack:
        socket_path = os.environ.get(TORCH_WORKER_SERVER_SOCKET)
        if socket_path is not None:
            stack.enter_context(_worker_server(socket_path))

        yield
