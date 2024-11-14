// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector flatten(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    auto data = inputs.at(0);
    auto axis = node.get_attribute_value<std::int64_t>("axis", 1);
    const auto data_rank = data.get_partial_shape().rank();

    if (data_rank.is_static()) {
        const std::int64_t data_rank_value = data_rank.get_length();
        // Accepted range is [-r, r] where r = rank(input).
        FRONT_END_GENERAL_CHECK(-data_rank_value <= axis && axis <= data_rank_value,
                                node.get_description(),
                                " axis ",
                                axis,
                                " out of tensor range [",
                                -data_rank_value,
                                ", ",
                                data_rank_value,
                                "]");
        axis = ov::util::normalize(axis, data_rank_value);
    }
    return {ov::op::util::flatten(data, static_cast<int>(axis))};
}

ONNX_OP("Flatten", OPSET_SINCE(1), ai_onnx::opset_1::flatten);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
