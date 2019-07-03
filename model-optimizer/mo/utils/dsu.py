"""
 Copyright (c) 2018-2019 Intel Corporation

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


class DSUElem:
    """
    An object that represents one DSU element.
    """
    name = ''
    parent = ''
    rank = 1

    def __init__(self, name):
        self.name = name
        self.parent = name
        self.rank = 1


class DSU:
    """
    Naive implementation of the "disjoint set union" data structure.
    """
    map = dict()

    def __init__(self, elems: list):
        self.map = {elem.name: elem for elem in elems}
        pass

    def find_elem(self, name: str):
        return self.map[name]

    def find_parent(self, elem: DSUElem):
        if elem.parent == elem.name:
            return elem
        parent_elem = self.find_parent(self.find_elem(elem.parent))
        elem.parent = parent_elem.name
        return parent_elem

    def union(self, elem1: DSUElem, elem2: DSUElem):
        elem1 = self.find_parent(elem1)
        elem2 = self.find_parent(elem2)
        if elem1.name == elem2.name:  # already in the same set
            return

        if elem1.rank < elem2.rank:
            elem1.parent = elem2.name
        elif elem1.rank > elem2.rank:
            elem2.parent = elem1.name
        else:
            elem1.parent = elem2.name
            elem2.rank = elem2.rank + 1
