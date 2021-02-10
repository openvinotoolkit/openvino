# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""
Package: ngraph.op
Low level wrappers for the nGraph c++ api in ngraph::op.
"""

# flake8: noqa

import numpy as np

from _pyngraph.op import Constant

"""Retrieve Constant inner data.

    Internally uses PyBind11 Numpy's buffer protocol.

    :return Numpy array containing internally stored constant data.
"""
Constant.get_data = lambda self: np.array(self, copy=True)

from _pyngraph.op import Parameter
