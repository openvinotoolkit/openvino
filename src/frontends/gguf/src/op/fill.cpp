// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_FILL sets every element of a tensor, or the range given by get_update_range, to a constant. The scalar is
// provided by the decoder as the "fill_value" attribute (ggml stores it as a float in op_params[0]). Broadcasting to
// ShapeOf(input) keeps the input live and works for dynamic shapes.
OutputVector translate_fill(const NodeContext& context) {
    num_inputs_check(context, 1, 3);

    auto x = context.get_input(0);
    float fill_value = context.get_attribute<float>("fill_value");

    // ggml FILL keeps the tensor's own type (f32 or f16), so do not hardcode f32 here.
    auto fill = [&](const ov::Output<ov::Node>& target) -> ov::Output<ov::Node> {
        auto val = ov::op::v0::Constant::create(target.get_element_type(), ov::Shape{}, {fill_value});
        return std::make_shared<ov::op::v3::Broadcast>(val,
                                                       std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64));
    };
    ov::Output<ov::Node> res;
    if (auto range = get_update_range(context)) {
        res = update_axis_range(x, range->first, range->second, context.get_attribute<int64_t>("axis", 2), fill);
    } else {
        res = fill(x);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
