"""Deterministic process workers; persistent pools and no nested pools."""
from contextlib import contextmanager
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle

import numpy as np
import torch


def start_method(requested="auto"):
    method = ("fork" if os.name == "posix" else "spawn") if requested == "auto" else requested
    if method not in mp.get_all_start_methods():
        raise ValueError(f"Unsupported multiprocessing start method: {method}")
    return method


@contextmanager
def worker_pool(workers, initializer, context, method="auto"):
    """Construct once per phase, reuse across checkpoint chunks."""
    if workers < 1:
        raise ValueError("Worker count must be positive")
    if workers == 1:
        initializer(context)
        yield None
        return
    if mp.current_process().daemon:
        raise RuntimeError("Nested process pools are forbidden; use one MC worker inside evaluation")
    pool = mp.get_context(start_method(method)).Pool(workers, initializer=initializer, initargs=(context,))
    try:
        yield pool
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()


def single_thread_worker():
    # Each process owns one CPU thread, as in the supplied parallel implementation.
    torch.set_num_threads(1)


def ordered_map(pool, function, items):
    return map(function, items) if pool is None else pool.imap(function, items, chunksize=1)


def atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def digest_arrays(arrays, metadata):
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode())
    for value in arrays:
        value = np.ascontiguousarray(value)
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(memoryview(value).cast("B"))
    return h.hexdigest()


def guard_checkpoint(directory, digest):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = directory / "fingerprint.json"
    if stamp.exists():
        if json.loads(stamp.read_text())["fingerprint"] != digest:
            raise RuntimeError(f"Checkpoint inputs differ; use a new output directory: {directory}")
    else:
        stamp.write_text(json.dumps({"fingerprint": digest}, indent=2))
    return directory
