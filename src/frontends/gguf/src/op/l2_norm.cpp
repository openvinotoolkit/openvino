// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// L2 normalization over the last dimension: x / max(sqrt(sum(x^2)), eps).
// op_case 100 (builder): clip the last dimension to an L2 norm threshold,
// x * clamp(threshold / (||x|| + eps), 0, 1). ReduceL2 keeps GPU execution out of F32 islands.
OutputVector translate_l2_norm(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    if (context.get_op_case() == 100) {
        const auto input = context.get_input(0);
        const auto rank = input.get_partial_shape().rank();
        FRONT_END_OP_CONVERSION_CHECK(rank.is_static(), "L2 clipping requires a static rank");
        ov::Output<ov::Node> norm = std::make_shared<ov::op::v4::ReduceL2>(
            input,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {rank.get_length() - 1}),
            true);
        const auto eps = context.get_attribute<float>("eps", 0.f);
        if (eps != 0.f) {
            norm = std::make_shared<ov::op::v1::Add>(norm, ov::op::v0::Constant::create(ov::element::f32, {}, {eps}));
        }
        const auto threshold =
            ov::op::v0::Constant::create(ov::element::f32, {}, {context.get_attribute<float>("threshold")});
        auto factor =
            std::make_shared<ov::op::v0::Clamp>(std::make_shared<ov::op::v1::Divide>(threshold, norm), 0., 1.);
        return rename_outputs_with_suffix({std::make_shared<ov::op::v1::Multiply>(input, factor)}, context.get_name());
    }

    return rename_outputs_with_suffix({make_l2_norm(context.get_input(0), context.get_attribute<float>("eps"))},
                                      context.get_name());
}

}  // namespace ov::frontend::gguf::op
