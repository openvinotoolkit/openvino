//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
