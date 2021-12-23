// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "ngraph/log.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

namespace py = pybind11;

void regclass_graph_op_If(py::module m);
