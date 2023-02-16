# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
