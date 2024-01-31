// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/flatten.hpp"

#include "exceptions.hpp"
#include "utils/ov_builders/reshape.hpp"
#include "validation_util.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
ov::OutputVector flatten(const Node& node) {
    ov::OutputVector inputs{node.get_ng_inputs()};
    auto data = inputs.at(0);
    auto axis = node.get_attribute_value<std::int64_t>("axis", 1);
    const auto data_rank = data.get_partial_shape().rank();

    if (data_rank.is_static()) {
        const std::int64_t data_rank_value = data_rank.get_length();
        // Accepted range is [-r, r] where r = rank(input).
        axis =
            ov::util::normalize_axis(node.get_description(), axis, data_rank_value, -data_rank_value, data_rank_value);
    }
    return {ov::op::util::flatten(data, static_cast<int>(axis))};
}

}  // namespace set_1

}  // namespace op

}  // namespace  onnx_import

}  // namespace  ngraph
