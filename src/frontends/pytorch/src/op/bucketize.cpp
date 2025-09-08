// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bucketize.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_bucketize(const NodeContext& context) {
    num_inputs_check(context, 2, 5);
    auto input = context.get_input(0);
    auto boundaries = context.get_input(1);

    element::Type output_type = ov::element::i64;
    if (!context.input_is_none(2) && context.const_input<bool>(2)) {
        output_type = ov::element::i32;
    } else if (context.has_attribute("out_int32") && context.get_attribute<bool>("out_int32")) {
        output_type = ov::element::i32;
    }

    bool with_right_bound = true;
    if (!context.input_is_none(3)) {
        with_right_bound = !context.const_input<bool>(3);
    } else if (context.has_attribute("right")) {
        with_right_bound = !context.get_attribute<bool>("right");
    }

    auto bucketize =
        context.mark_node(std::make_shared<v3::Bucketize>(input, boundaries, output_type, with_right_bound));

    if (!context.input_is_none(4)) {
        context.mutate_input(4, bucketize);
    }

    return {bucketize};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov