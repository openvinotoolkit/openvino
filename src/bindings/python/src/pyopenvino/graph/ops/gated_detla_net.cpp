// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/gated_delta_net.hpp"

#include "openvino/op/op.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_graph_op_GatedDeltaNet(py::module m) {
    using ov::op::GatedDeltaNet;
    py::class_<GatedDeltaNet, std::shared_ptr<GatedDeltaNet>, ov::Node> cls(
        m,
        "_GatedDeltaNet");
    cls.doc() = "Experimental extention for GatedDeltaNet operation. Use with care: no backward compatibility is "
                "guaranteed in future releases.";
    cls.def(py::init<const ov::OutputVector&>());
}
