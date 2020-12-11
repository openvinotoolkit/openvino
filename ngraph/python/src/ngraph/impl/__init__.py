# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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
Package: ngraph
Low level wrappers for the nGraph c++ api.
"""

# flake8: noqa

import os
import sys

if sys.platform == "win32":
    # ngraph.dll is installed 3 directories above by default
    # and this path needs to be visible to the ngraph.pyngraph module
    #
    # If you're using a custom installation of nGraph,
    # add the location of ngraph.dll to your system PATH.
    ngraph_dll = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    os.environ["PATH"] = os.path.abspath(ngraph_dll) + ";" + os.environ["PATH"]

from ngraph.pyngraph import Dimension
from ngraph.pyngraph import Function
from ngraph.pyngraph import Input
from ngraph.pyngraph import Output
from ngraph.pyngraph import Node
from ngraph.pyngraph import Type
from ngraph.pyngraph import PartialShape
from ngraph.pyngraph import Shape
from ngraph.pyngraph import Strides
from ngraph.pyngraph import CoordinateDiff
from ngraph.pyngraph import AxisSet
from ngraph.pyngraph import AxisVector
from ngraph.pyngraph import Coordinate

from ngraph.pyngraph import util
