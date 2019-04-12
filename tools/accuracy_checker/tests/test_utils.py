"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from accuracy_checker.utils import concat_lists, contains_all, contains_any, overrides, zipped_transform


def test_concat_lists():
    assert ['a', 'b'] == concat_lists(['a'], ['b'])
    assert ['a', 'b', 'c'] == concat_lists(['a'], ['b'], ['c'])
    assert ['a', 'b', 'c'] == concat_lists(['a', 'b'], ['c'])
    assert ['a'] == concat_lists(['a'], [])
    assert [] == concat_lists([], [], [])
    assert [] == concat_lists([])


def test_contains_all():
    assert contains_all([1, 2, 3], [1, 2])
    assert contains_all([1, 2, 3], [1, 2], [3])
    assert not contains_all([1, 2, 3], [1, 5])


def test_contains_any():
    assert contains_any([1, 2, 3], [1])
    assert contains_any([1, 2, 3], [4, 5, 2])
    assert not contains_any([1, 2, 3], [4, 5])


class TestZippedTransform:
    def test_two_iterables(self):
        as_ = [2, 3, 5]
        bs = [2, 3, 6]

        ras, rbs = zipped_transform(lambda a, b: (a + b, a - b), as_, bs)

        assert ras == [4, 6, 11]
        assert rbs == [0, 0, -1]
        assert as_ == [2, 3, 5]
        assert bs == [2, 3, 6]

    def test_inplace(self):
        as_ = [2, 3, 5]
        bs = [2, 3, 6]

        zipped_transform(lambda a, b: (a + b, a - b), as_, bs, inplace=True)

        assert as_ == [4, 6, 11]
        assert bs == [0, 0, -1]

    def test_three_iterables(self):
        as_ = [1, 1, 1]
        bs = [2, 2, 2]
        cs = [3, 3, 3]

        ras, rbs, rcs = zipped_transform(lambda a, b, c: (a + 1, b + 2, c + 3), as_, bs, cs)

        assert ras == [2, 2, 2]
        assert rbs == [4, 4, 4]
        assert rcs == [6, 6, 6]

    def test_none_function(self):
        xs = [1, 1, 1]
        ys = [1, 1, 1]
        zipped_transform(lambda a, b: None, xs, ys)


class TestOverrides:
    def test_negative(self):
        class A:
            def foo(self):
                pass

        class B(A):
            pass

        assert not overrides(B, 'foo')
        assert not overrides(B(), 'foo')

    def test_positive(self):
        class A:
            def foo(self):
                pass

        class B(A):
            def foo(self):
                pass

        assert overrides(B, 'foo')
        assert overrides(B(), 'foo')

    def test_three_class(self):
        class A:
            def foo(self): pass

        class B(A):
            pass

        class C(B):
            def foo(self): pass

        assert overrides(C, 'foo')
        assert not overrides(B, 'foo')

    def test_custom_base(self):
        class A:
            def foo(self): pass

        class B:
            def foo(self): pass

        class C:
            pass

        assert overrides(B, 'foo', A)
        assert not overrides(C, 'foo', A)
