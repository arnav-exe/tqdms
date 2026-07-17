"""
Thin wrappers around `concurrent.futures`.
"""
from contextlib import contextmanager
from operator import length_hint

from ..auto import tqdm as tqdm_auto
from ..std import TqdmWarning

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['thread_map', 'process_map', 'interpreter_map']


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


def _min_map_len(iterables):
    """min(map(length_hint, iterables))"""
    return min(n for it in iterables if (n := length_hint(it, -1)) >= 0)


def _executor_map(
    PoolExecutor, fn, *iterables, max_workers=None, timeout=None, chunksize=1, lock_name="",
    tqdm_class=tqdm_auto, smoothing=0.0, _share_lock=True, **tqdm_kwargs
):
    """
    Implementation of `thread_map`, `process_map` and `interpreter_map`.

    Parameters
    ----------
    max_workers  : int
    timeout  : int
    buffersize  : int
        Requires Python>=3.14.
    thread_name_prefix  : str
    max_tasks_per_child  : int
    mp_context  : str
    """
    kwargs = tqdm_kwargs.copy()
    if 'total' not in kwargs:
        kwargs['total'] = _min_map_len(iterables)
    map_kwargs = {}
    if 'buffersize' in kwargs:
        map_kwargs['buffersize'] = kwargs.pop('buffersize')
    pool_kwargs = {}
    for k in ('thread_name_prefix', 'max_tasks_per_child', 'mp_context'):
        if k in kwargs:
            pool_kwargs[k] = kwargs.pop(k)
    dynamic_miniters = None
    if kwargs['total'] and 'miniters' not in kwargs:
        try:
            from os import process_cpu_count as cpu_count
        except ImportError:
            from os import cpu_count
        # thread & process pools have different default workers, but we KISS here
        rough_max = max_workers or min(32, (cpu_count() or 1) + 4)
        if kwargs['total'] > rough_max:
            kwargs['miniters'] = rough_max
            dynamic_miniters = True
    with ensure_lock(tqdm_class, lock_name=lock_name) as lk:
        if _share_lock:
            # share lock in case workers are already using `tqdm`
            pool_kwargs.update(initializer=tqdm_class.set_lock, initargs=(lk,))
        with PoolExecutor(max_workers=max_workers, **pool_kwargs) as ex:
            with tqdm_class(smoothing=smoothing, **kwargs) as pbar:
                if dynamic_miniters is not None:
                    pbar.dynamic_miniters = True
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
        Maximum number of workers to spawn; passed to `concurrent.futures.ThreadPoolExecutor`.
    thread_name_prefix  : str, optional
        Passed to `concurrent.futures.ThreadPoolExecutor` [default: ''].
    timeout  : int or float, optional
        Seconds to wait before raising `TimeoutError` if `__next__` is called and the
        result isn't available. [default: None].
    buffersize  : int, optional
        Requires Python>=3.14 [default: None].
    tqdm_class  : optional
        `tqdm` class to use for bars [default: tqdm.auto.tqdm].
    smoothing  : float, optional
        Passed to `tqdm_class`; the [default: 0] is average (due to erratic update frequency).
    lock_name  : str, optional
        Member of `tqdm_class.get_lock()` to use [default: ''].
    """
    from concurrent.futures import ThreadPoolExecutor
    return _executor_map(ThreadPoolExecutor, fn, *iterables, **tqdm_kwargs)


def interpreter_map(fn, *iterables, **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.InterpreterPoolExecutor` (Python 3.14+).

    Parameters
    ----------
    Same as `thread_map`.

    Notes
    -----
    `fn`, its arguments, and its return values must be pickleable.
    """
    from concurrent.futures import InterpreterPoolExecutor
    return _executor_map(InterpreterPoolExecutor, fn, *iterables, _share_lock=False,
                         **tqdm_kwargs)


def process_map(fn, *iterables, lock_name="mp_lock", **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to `concurrent.futures.ProcessPoolExecutor`.
    timeout  : int or float, optional
        Seconds to wait before raising `TimeoutError` if `__next__` is called and the
        result isn't available. [default: None].
    chunksize  : int, optional
        Approximate size of chunks sent to worker processes; passed to
        `concurrent.futures.ProcessPoolExecutor.map`. [default: 1].
    buffersize  : int, optional
        Requires Python>=3.14 [default: None].
    max_tasks_per_child  : int, optional
        Maximum number of tasks a worker process can complete before being replaced
        with a new process; passed to `concurrent.futures.ProcessPoolExecutor`.
    mp_context  : multiprocessing.BaseContext, optional
        Multiprocessing context to use, e.g. `multiprocessing.get_context('fork')`.
    lock_name  : str, optional
        Member of `tqdm_class.get_lock()` to use [default: mp_lock].
    tqdm_class  : optional
        `tqdm` class to use for bars [default: tqdm.auto.tqdm].
    smoothing  : float, optional
        Passed to `tqdm_class`; the [default: 0] is average (due to erratic update frequency).
    """
    from concurrent.futures import ProcessPoolExecutor
    if iterables and 'chunksize' not in tqdm_kwargs:
        # default `chunksize=1` has poor performance for large iterables
        # (most time spent dispatching items to workers).
        shortest_iterable_len = _min_map_len(iterables)
        if shortest_iterable_len > 1000:
            from warnings import warn
            warn("Iterable length %d > 1000 but `chunksize` is not set."
                 " This may seriously degrade multiprocess performance."
                 " Set `chunksize=1` or more." % shortest_iterable_len,
                 TqdmWarning, stacklevel=2)
    return _executor_map(ProcessPoolExecutor, fn, *iterables, lock_name=lock_name, **tqdm_kwargs)
