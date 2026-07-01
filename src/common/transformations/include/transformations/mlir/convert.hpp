// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/util/env_util.hpp"
#include "transformations_visibility.hpp"

namespace ov {

namespace pass {

inline bool is_mlir_transform_enabled() {
    return !util::getenv_string("OV_MLIR_MODE").empty() && util::getenv_bool("OV_MLIR", true);
}

void TRANSFORMATIONS_API transformMLIR(std::shared_ptr<ov::Model> model,
                                       std::shared_ptr<ov::EvaluationContext> loweringContext);

}

namespace mlir {

// Resolves dynamic dims of MLIROp output shapes from runtime input shapes via
// the op's dimensions_map. Asserts if the node is not an MLIROp.
// Exposed here so callers outside transformations (e.g. Intel GPU plugin) can
// invoke shape inference without depending on the private MLIROp header.
std::vector<ov::PartialShape> TRANSFORMATIONS_API mlir_op_shape_infer(
        const std::shared_ptr<const ov::Node>& op,
        const std::vector<ov::PartialShape>& input_shapes);

}
}
