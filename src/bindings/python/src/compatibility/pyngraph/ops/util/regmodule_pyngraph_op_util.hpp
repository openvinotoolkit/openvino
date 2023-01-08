// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include "pyngraph/ops/util/arithmetic_reduction.hpp"
#include "pyngraph/ops/util/binary_elementwise_arithmetic.hpp"
#include "pyngraph/ops/util/binary_elementwise_comparison.hpp"
#include "pyngraph/ops/util/binary_elementwise_logical.hpp"
#include "pyngraph/ops/util/index_reduction.hpp"
#include "pyngraph/ops/util/op_annotations.hpp"
#include "pyngraph/ops/util/unary_elementwise_arithmetic.hpp"

namespace py = pybind11;

void regmodule_pyngraph_op_util(py::module m);
