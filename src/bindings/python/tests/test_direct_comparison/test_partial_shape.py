import pytest
from openvino import PartialShape

def test_partial_shape_eq_list():
    a = [1, 2, 3]
    ps = PartialShape(a)
    assert a == ps, "The PartialShape object is not equal to the list"

def test_partial_shape_eq_tuple():
    t = (1, 2, 3)
    ps = PartialShape(t)
    assert t == ps, "The PartialShape object is not equal to the tuple"

if __name__ == "__main__":
    pytest.main()
