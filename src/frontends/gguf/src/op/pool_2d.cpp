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

OutputVector translate_pool_2d(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    const auto params = context.get_attribute<std::vector<int32_t>>("pool_params");
    FRONT_END_OP_CONVERSION_CHECK(params.size() == 6, "POOL_2D requires 6 params");

    const auto kernel_width = static_cast<size_t>(params[0]);
    const auto kernel_height = static_cast<size_t>(params[1]);
    const auto stride_x = static_cast<size_t>(params[2]);
    const auto stride_y = static_cast<size_t>(params[3]);
    const auto padding_x = static_cast<size_t>(params[4]);
    const auto padding_y = static_cast<size_t>(params[5]);

    const ov::Shape kernel{kernel_height, kernel_width};
    const ov::Strides strides{stride_y, stride_x};
    const ov::Shape pads_begin{padding_y, padding_x};
    const ov::Shape pads_end = pads_begin;

    auto input = context.get_input(0);
    if (input.get_element_type() != ov::element::f32) {
        input = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    }

    ov::Output<Node> result;
    switch (context.get_op_case()) {
    case 1:
        result = std::make_shared<ov::op::v1::MaxPool>(input, strides, pads_begin, pads_end, kernel);
        break;
    case 2:
        result = std::make_shared<ov::op::v1::AvgPool>(input, strides, pads_begin, pads_end, kernel, false);
        break;
    default:
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported POOL_2D mode");
    }

    return rename_outputs_with_suffix({std::move(result)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
