// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/op/util/sub_graph_base.hpp"

namespace py = pybind11;

using MultiSubgraphInputDescriptionVector = ov::op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector;
using MultiSubgraphOutputDescriptionVector = ov::op::util::MultiSubGraphOp::MultiSubgraphOutputDescriptionVector;
namespace MultiSubgraphHelpers {
bool is_constant_or_parameter(const std::shared_ptr<ov::Node>& node);
MultiSubgraphInputDescriptionVector list_to_input_descriptor(const py::list& inputs);
MultiSubgraphOutputDescriptionVector list_to_output_descriptor(const py::list& outputs);
}  // namespace MultiSubgraphHelpers

void regclass_graph_op_util_MultiSubgraphOp(py::module m);
