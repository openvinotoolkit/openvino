"""
 Copyright (C) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from .cimport test_utils_api_impl_defs as C
from ..inference_engine.ie_api cimport IENetwork

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.pair cimport pair

def CompareNetworks(IENetwork lhs, IENetwork rhs):
    cdef pair[bool, string] c_pair
    c_pair = C.CompareNetworks(lhs.impl, rhs.impl)
    return c_pair
