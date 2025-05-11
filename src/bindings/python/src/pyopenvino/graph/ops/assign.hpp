// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include "openvino/op/assign.hpp"

namespace py = pybind11;

void regclass_graph_op_Assign(py::module m);
