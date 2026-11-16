// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_POOL_1D pools along the innermost axis with "pool_params" {mode, kernel, stride, padding}.
OutputVector translate_pool_1d(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    const auto params = context.get_attribute<std::vector<int32_t>>("pool_params");
    const auto mode = context.get_attribute<std::string>("pool_mode", "max");
    FRONT_END_OP_CONVERSION_CHECK(params.size() == 4, "POOL_1D requires 4 params");
    FRONT_END_OP_CONVERSION_CHECK((params[0] == 0 && mode == "max") || (params[0] == 1 && mode == "average"),
                                  "POOL_1D pool_mode does not match pool_params");
    FRONT_END_OP_CONVERSION_CHECK(params[1] > 0 && params[2] > 0, "POOL_1D kernel and stride must be positive");
    FRONT_END_OP_CONVERSION_CHECK(params[3] >= 0 && params[3] < params[1],
                                  "POOL_1D padding must be nonnegative and smaller than the kernel");

    auto input = context.get_input(0);
    const auto rank = input.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(rank.is_static() && rank.get_length() >= 1 && rank.get_length() <= 5,
                                  "POOL_1D requires an input of rank 1 to 5");
    if (input.get_element_type() != ov::element::f32) {
        input = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    }
    // OpenVINO pools need batch and channel axes in front of the spatial ones.
    const int64_t added = std::max<int64_t>(0, 3 - rank.get_length());
    std::vector<int64_t> added_axes(static_cast<size_t>(added));
    std::iota(added_axes.begin(), added_axes.end(), 0);
    const auto added_axes_node = ov::op::v0::Constant::create(ov::element::i64, {added_axes.size()}, added_axes);
    if (added > 0) {
        input = std::make_shared<ov::op::v0::Unsqueeze>(input, added_axes_node);
    }

    const size_t spatial = static_cast<size_t>(rank.get_length() + added - 2);
    ov::Shape kernel(spatial, 1);
    ov::Strides strides(spatial, 1);
    ov::Shape pads(spatial, 0);
    kernel.back() = static_cast<size_t>(params[1]);
    strides.back() = static_cast<size_t>(params[2]);
    pads.back() = static_cast<size_t>(params[3]);

    ov::Output<Node> result;
    if (mode == "max") {
        result = std::make_shared<ov::op::v1::MaxPool>(input, strides, pads, pads, kernel);
    } else {
        // ggml averages over the cells inside the input only.
        result = std::make_shared<ov::op::v1::AvgPool>(input, strides, pads, pads, kernel, true);
    }
    if (added > 0) {
        result = std::make_shared<ov::op::v0::Squeeze>(result, added_axes_node);
    }

    return rename_outputs_with_suffix({std::move(result)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
