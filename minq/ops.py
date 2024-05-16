from functools import reduce
from typing import Any, cast

from minq.utils import from_binary

import numpy as np
import numpy.typing as npt

def ket(*args: int | str):
    '''Creates a standard basis vector'''
    if len(args) == 0:
        return np.array([1])
    elif isinstance(args[0], int):
        assert all(b == 0 or b == 1 for b in args)
        x = np.zeros(2 ** len(args))
        x[from_binary(*cast(tuple[int, ...], args))] = 1
        return x
    else:
        assert len(args) == 1
        x = np.zeros(2 ** len(args[0]))
        x[int(args[0], base=2)] = 1
        return x

def kron(*arrays: npt.ArrayLike) -> npt.NDArray[Any]:
    '''Computes the Kronecker product (tensor product) of the given arrays'''
    assert len(arrays) > 0
    return reduce(np.kron, arrays)

def kronpow(array: npt.ArrayLike, n: int) -> npt.NDArray[Any]:
    '''Applies the Kronecker product n times'''
    assert n > 0
    return reduce(lambda acc, _: np.kron(acc, array), range(n - 1), array)
