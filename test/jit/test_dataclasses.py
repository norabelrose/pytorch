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

class TestDataclasses(JitTestCase):
    def test_init(self):
        @dataclass
        class Point(object):
            x: Tensor
            y: Tensor
            # norm: Optional[Tensor]

            def __post_init__(self):
                print((self.x ** 2 + self.y ** 2) ** 0.5)
            
            def __add__(self, other: 'Point') -> 'Point':
                return Point(self.x + other.x, self.y + other.y)
            
            def __mul__(self, scalar: float) -> 'Point':
                return Point(self.x * scalar, self.y * scalar)

        def fn(x, y):
            pt1 = Point(x, y)
            pt2 = pt1 * 42.0
            return (pt1 + pt2)

        x = torch.tensor(2.0)
        y = torch.tensor(3.0)
        self.checkScript(fn, [x, y], optimize=True)