// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/axes_util.hpp"

#include "utils.hpp"
#include "validation_util.hpp"

namespace ov {
namespace op {
namespace util {
AxisSet get_normalized_axes_from_tensor(const Node* const node, const Tensor& tensor, const Rank& rank) {
    const auto axes = ov::get_tensor_data_as<int64_t>(tensor);

    return {ov::util::normalize_axes(node->get_friendly_name(), axes, rank)};
}
}  // namespace util
}  // namespace op
}  // namespace ov
