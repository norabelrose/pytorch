# Owner(s): ["oncall: jit"]

from dataclasses import dataclass, field
from hypothesis import given, strategies as st
from torch.testing._internal.jit_utils import JitTestCase
from typing import Optional
import torch

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# Example jittable dataclass
@dataclass(order=True)
class Point:
    x: float
    y: float
    norm: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.norm = (torch.tensor(self.x) ** 2 + torch.tensor(self.y) ** 2) ** 0.5


# Hypothesis strategies
ExtendedReals = st.floats(allow_infinity=True, allow_nan=False)

class TestDataclasses(JitTestCase):
    # Sort of tests both __post_init__ and optional fields
    @given(ExtendedReals, ExtendedReals)
    def test__post_init__(self, x, y):
        def fn(x: float, y: float):
            pt = Point(x, y)
            return pt.norm

        self.checkScript(fn, [x, y])

    @given(st.tuples(ExtendedReals, ExtendedReals), st.tuples(ExtendedReals, ExtendedReals))
    def test_comparators(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2

        def compare(x1: float, y1: float, x2: float, y2: float):
            pt1 = Point(x1, y1)
            pt2 = Point(x2, y2)
            return (
                pt1 == pt2,
                # pt1 != pt2,   # TODO: Modify interpreter to auto-resolve (a != b) to not (a == b) when there's no __ne__
                pt1 < pt2,
                pt1 <= pt2,
                pt1 > pt2,
                pt1 >= pt2,
            )

        self.checkScript(compare, [x1, y1, x2, y2])

    def test_default_factories(self):
        @dataclass
        class Foo(object):
            x: list[int] = field(default_factory=list)

        with self.assertRaises(NotImplementedError):
            def fn():
                foo = Foo()
                return foo.x

            torch.jit.script(fn)()

    # The user should be able to write their own __eq__ implementation
    # without us overriding it.
    def test_custom__eq__(self):
        @dataclass
        class CustomEq:
            a: int
            b: int

            def __eq__(self, other: 'CustomEq') -> bool:
                return self.a == other.a  # ignore the b field

        def fn(a: int, b1: int, b2: int):
            pt1 = CustomEq(a, b1)
            pt2 = CustomEq(a, b2)
            return pt1 == pt2

        self.checkScript(fn, [1, 2, 3])
