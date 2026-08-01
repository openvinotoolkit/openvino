// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/core/node_output.hpp"
#include "openvino/decompositions/rms_norm.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/node_registry.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_rms_norm(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    // ggml RMS_NORM is norm-only (the weight/gamma multiply is a separate ggml MUL op), so no scale
    // is passed. The shared helper emits the canonical form ov::pass::RMSFusion folds into
    // ov::op::internal::RMS.
    auto input_node = context.get_input(0);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto eps = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {context.get_attribute<float>("eps")});

    ov::pass::NodeRegistry reg;
    auto res = ov::decomposition::rms_norm(reg, input_node, axes, eps);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
