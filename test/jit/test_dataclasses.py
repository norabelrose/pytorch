# Owner(s): ["oncall: jit"]

from dataclasses import dataclass
from typing import Optional
from torch import Tensor
from torch.testing._internal.jit_utils import JitTestCase
import torch

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# Example jittable dataclass
@dataclass(order=True)
class Point(object):
    x: Tensor
    y: Tensor
    # norm: Optional[Tensor]

    def __post_init__(self):
        pass # print((self.x ** 2 + self.y ** 2) ** 0.5)
    
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)


class TestDataclasses(JitTestCase):
    def test_init(self):
        def fn(x, y):
            pt1 = Point(x, y)
            pt2 = pt1 * 42.0
            return (pt1 + pt2)

        x = torch.tensor(2.0)
        y = torch.tensor(3.0)
        self.checkScript(fn, [x, y])
    
    def test_hash(self):
        x = torch.tensor(2.0)
        y = torch.tensor(3.0)
        with self.assertRaises(RuntimeError):
            def hash_fn(x, y):
                return hash(Point(x, y))
            torch.jit.script(hash_fn)(x, y)
    
    # def test_repr(self):
    #     def fn(x, y):
    #         pt = Point(x, y)
    #         return repr(pt)
# 
    #     x = torch.tensor(2.0)
    #     y = torch.tensor(3.0)
    #     self.checkScript(fn, [x, y], optimize=True)
    
    def test_comparators(self):
        x = torch.tensor(2.0)
        y = torch.tensor(3.0)
        def eq(x, y):
            return Point(x, y) == Point(x, y)
        self.checkScript(eq, [x, y])
        
        # def ne(x, y):
        #     return Point(x, y) != Point(x, y)
        # self.checkScript(ne, [x, y], optimize=True)
        
        def lt(x, y):
            return Point(x, y) < Point(x, y) * 2.0
        self.checkScript(lt, [x, y])
        
        def gt(x, y):
            return Point(x, y) * 2.0 > Point(x, y)
        self.checkScript(gt, [x, y])
        
        def le(x, y):
            return Point(x, y) <= Point(x, y) * 2.0
        self.checkScript(le, [x, y])
        
        def ge(x, y):
            return Point(x, y) * 2.0 >= Point(x, y)
        self.checkScript(ge, [x, y])
