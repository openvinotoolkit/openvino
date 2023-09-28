// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/axes_util.hpp"

#include "openvino/core/validation_util.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {
AxisSet get_normalized_axes_from_tensor(const Node* const node, const Tensor& tensor, const Rank& rank) {
    const auto axes = ov::get_tensor_data_as<int64_t>(tensor);

    OPENVINO_SUPPRESS_DEPRECATED_START
    return {normalize_axes(node->get_friendly_name(), axes, rank)};
    OPENVINO_SUPPRESS_DEPRECATED_END
}
}  // namespace util
}  // namespace op
}  // namespace ov
