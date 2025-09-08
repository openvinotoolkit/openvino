// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "pyopenvino/graph/ops/util/arithmetic_reduction.hpp"
#include "pyopenvino/graph/ops/util/binary_elementwise_arithmetic.hpp"
#include "pyopenvino/graph/ops/util/binary_elementwise_comparison.hpp"
#include "pyopenvino/graph/ops/util/binary_elementwise_logical.hpp"
#include "pyopenvino/graph/ops/util/index_reduction.hpp"
#include "pyopenvino/graph/ops/util/unary_elementwise_arithmetic.hpp"
#include "pyopenvino/graph/ops/util/variable.hpp"
#include "pyopenvino/graph/ops/util/multisubgraph.hpp"

namespace py = pybind11;

void regmodule_graph_op_util(py::module m);
