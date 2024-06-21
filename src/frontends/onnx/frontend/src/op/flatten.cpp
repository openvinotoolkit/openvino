// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/validation_util.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector flatten(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
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

static bool registered = register_translator("Flatten", VersionRange::single_version_for_all_opsets(), flatten);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
