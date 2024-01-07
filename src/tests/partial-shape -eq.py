class Shape:
    # ... other methods and attributes ...

    def __eq__(self, other):
        if isinstance(other, Shape):
            return self.dims == other.dims
        elif isinstance(other, (list, tuple)):
            return self.dims == list(other)
        return NotImplemented

class PartialShape:
    # ... other methods and attributes ...

    def __eq__(self, other):
        if isinstance(other, PartialShape):
            return self.dims == other.dims
        elif isinstance(other, (list, tuple)):
            return self.dims == list(other)
        return NotImplemented

import unittest
from module import Shape, PartialShape

class TestShapeComparison(unittest.TestCase):
    def test_shape_eq_list(self):
        a = [1, 2, 3]
        s = Shape(a)
        self.assertEqual(a, s)

    def test_partial_shape_eq_tuple(self):
        t = (1, 2, 3)
        ps = PartialShape(t)
        self.assertEqual(t, ps)


if __name__ == '__main__':
    unittest.main()
