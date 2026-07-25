"""
Thin wrappers around `concurrent.futures`.
"""
from contextlib import contextmanager
from operator import length_hint
from os import cpu_count

from ..auto import tqdm as tqdm_auto
from ..std import TqdmWarning

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['thread_map', 'process_map']


@contextmanager
def ensure_lock(tqdm_class, lock_name=""):
    """get (create if necessary) and then restore `tqdm_class`'s lock"""
    old_lock = getattr(tqdm_class, '_lock', None)  # don't create a new lock
    lock = old_lock or tqdm_class.get_lock()  # maybe create a new lock
    lock = getattr(lock, lock_name, lock)  # maybe subtype
    tqdm_class.set_lock(lock)
    yield lock
    if old_lock is None:
        del tqdm_class._lock
    else:
        tqdm_class.set_lock(old_lock)


def _executor_map(
    PoolExecutor, fn, *iterables, timeout=None, chunksize=1, lock_name="",
    tqdm_class=tqdm_auto, **tqdm_kwargs
):
    """
    Implementation of `thread_map` and `process_map`.

    Parameters
    ----------
    buffersize  : Requires Python>=3.14 [default: None].
    max_workers  : [default: min(32, cpu_count() + 4)].
    """
    kwargs = tqdm_kwargs.copy()
    if "total" not in kwargs:
        kwargs["total"] = length_hint(iterables[0])
    map_kwargs = {}
    if 'buffersize' in kwargs:
        map_kwargs['buffersize'] = kwargs.pop('buffersize')
    max_workers = kwargs.pop("max_workers", min(32, cpu_count() + 4))
    with ensure_lock(tqdm_class, lock_name=lock_name) as lk:
        # share lock in case workers are already using `tqdm`
        with PoolExecutor(max_workers=max_workers, initializer=tqdm_class.set_lock,
                          initargs=(lk,)) as ex:
            with tqdm_class(**kwargs) as pbar:
                orisubmit = ex.submit

                def patchsubmit(*args, **kwargs):
                    fut = orisubmit(*args, **kwargs)
                    fut.add_done_callback(lambda _: pbar.update())
                    return fut
                ex.submit = patchsubmit
                return list(ex.map(
                    fn, *iterables, timeout=timeout, chunksize=chunksize, **map_kwargs))


def thread_map(fn, *iterables, **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ThreadPoolExecutor`.

    Parameters
    ----------
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to
        `concurrent.futures.ThreadPoolExecutor.__init__`.
        [default: max(32, cpu_count() + 4)].
    timeout  : int or float, optional
        Seconds to wait before raising `TimeoutError` if `__next__` is called and the
        result isn't available. [default: None].
    tqdm_class  : optional
        `tqdm` class to use for bars [default: tqdm.auto.tqdm].
    lock_name  : str, optional
        Member of `tqdm_class.get_lock()` to use [default: mp_lock].
    """
    from concurrent.futures import ThreadPoolExecutor
    return _executor_map(ThreadPoolExecutor, fn, *iterables, **tqdm_kwargs)


def process_map(fn, *iterables, **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to
        `concurrent.futures.ProcessPoolExecutor.__init__`.
        [default: min(32, cpu_count() + 4)].
    timeout  : int or float, optional
        Seconds to wait before raising `TimeoutError` if `__next__` is called and the
        result isn't available. [default: None].
    chunksize  : int, optional
        Approximate size of chunks sent to worker processes; passed to
        `concurrent.futures.ProcessPoolExecutor.map`. [default: 1].
    tqdm_class  : optional
        `tqdm` class to use for bars [default: tqdm.auto.tqdm].
    lock_name  : str, optional
        Member of `tqdm_class.get_lock()` to use [default: mp_lock].
    """
    from concurrent.futures import ProcessPoolExecutor
    if iterables and "chunksize" not in tqdm_kwargs:
        # default `chunksize=1` has poor performance for large iterables
        # (most time spent dispatching items to workers).
        longest_iterable_len = max(map(length_hint, iterables))
        if longest_iterable_len > 1000:
            from warnings import warn
            warn("Iterable length %d > 1000 but `chunksize` is not set."
                 " This may seriously degrade multiprocess performance."
                 " Set `chunksize=1` or more." % longest_iterable_len,
                 TqdmWarning, stacklevel=2)
    if "lock_name" not in tqdm_kwargs:
        tqdm_kwargs = tqdm_kwargs.copy()
        tqdm_kwargs["lock_name"] = "mp_lock"
    return _executor_map(ProcessPoolExecutor, fn, *iterables, **tqdm_kwargs)
