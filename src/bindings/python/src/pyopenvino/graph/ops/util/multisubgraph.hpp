// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "ngraph/log.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

namespace py = pybind11;

using MultiSubgraphInputDescriptionVector = ov::op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector;
using MultiSubgraphOutputDescriptionVector = ov::op::util::MultiSubGraphOp::MultiSubgraphOutputDescriptionVector;
namespace MultiSubgraphOp {
MultiSubgraphInputDescriptionVector list_to_input_descriptor(const py::list& inputs);
MultiSubgraphOutputDescriptionVector list_to_output_descriptor(py::list& outputs);
}  // namespace MultiSubgraphOp

void regclass_graph_op_util_MultiSubgraphOp(py::module m);
