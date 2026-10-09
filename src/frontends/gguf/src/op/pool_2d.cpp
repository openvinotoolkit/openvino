// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/max_pool.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// ggml_pool_2d over [W, H, C, N] (OV [N, C, H, W], i.e. NCHW). "pool_params" carries ggml's op_params
// {op, k0, k1, s0, s1, p0, p1}: op is ggml_op_pool (0 = MAX, 1 = AVG) and the 0/1 suffixes are the
// width/height axes. ggml pads symmetrically and rounds the output size down; average pooling divides by
// the full kernel area, counting padded positions as zeros.
OutputVector translate_pool_2d(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    const auto params = context.get_attribute<std::vector<int32_t>>("pool_params");
    FRONT_END_OP_CONVERSION_CHECK(params.size() == 7, "POOL_2D requires {op, k0, k1, s0, s1, p0, p1}");
    FRONT_END_OP_CONVERSION_CHECK(params[1] > 0 && params[2] > 0, "POOL_2D kernel dimensions must be positive");
    FRONT_END_OP_CONVERSION_CHECK(params[3] > 0 && params[4] > 0, "POOL_2D strides must be positive");
    FRONT_END_OP_CONVERSION_CHECK(params[5] >= 0 && params[6] >= 0, "POOL_2D padding must be nonnegative");

    const auto u = [](int32_t value) {
        return static_cast<size_t>(value);
    };
    const ov::Shape kernel{u(params[2]), u(params[1])};
    const ov::Strides strides{u(params[4]), u(params[3])};
    const ov::Shape pads{u(params[6]), u(params[5])};

    auto input = context.get_input(0);
    if (input.get_element_type() != ov::element::f32) {
        input = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    }

    ov::Output<Node> result;
    switch (params[0]) {
    case 0:
        result = std::make_shared<ov::op::v1::MaxPool>(input, strides, pads, pads, kernel);
        break;
    case 1:
        result = std::make_shared<ov::op::v1::AvgPool>(input, strides, pads, pads, kernel, false);
        break;
    default:
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported POOL_2D mode ", params[0]);
    }

    return rename_outputs_with_suffix({std::move(result)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
