// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/core/node.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

namespace py = pybind11;

void regclass_graph_op_Loop(py::module m);