// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/op/constant.hpp"

namespace py = pybind11;

void regclass_graph_op_Constant(py::module m);
