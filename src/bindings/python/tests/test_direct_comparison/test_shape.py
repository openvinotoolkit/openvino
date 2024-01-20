from openvino import Shape, PartialShape

def test_shape_eq_list():
    a = [1, 2, 3]
    s = Shape(a)
    assert a == s

def test_shape_eq_tuple():
    t = (1, 2, 3)
    s = Shape(t)
    assert t == s

def test_partial_shape_eq_list():
    a = [1, 2, 3]
    ps = PartialShape(a)
    assert a == ps

def test_partial_shape_eq_tuple():
    t = (1, 2, 3)
    ps = PartialShape(t)
    assert t == ps
