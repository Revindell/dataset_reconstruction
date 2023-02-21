import multiprocessing.pool as mpp
from typing import Any, Callable, Iterable


def istarmap(self, func: Callable, iterable: Iterable, chunksize: int = 1) -> Any:
    """ Starmap-version of imap. Source: https://tutorialmeta.com/question/starmap-combined-with-tqdm
    Allows to use starmap with tqdm package to visualize a progress bar.

    :param self:
    :param func: function to be applied on multiple processors
    :param iterable: iterable which is given to the function
    :param chunksize: how many values shall be returned
    :return: return of the function given

    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


# apply created patch adding to Pool
mpp.Pool.istarmap = istarmap
