// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/evaluate_helpers.hpp"

#include "openvino/op/util/evaluate_helpers.hpp"

namespace ngraph {
AxisSet get_normalized_axes_from_tensor(const HostTensorPtr tensor,
                                        const ngraph::Rank& rank,
                                        const std::string& node_description) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto axes_vector = host_tensor_2_vector<int64_t>(tensor);
    const auto normalized_axes = ngraph::normalize_axes(node_description, axes_vector, rank);
    OPENVINO_SUPPRESS_DEPRECATED_END
    return AxisSet{normalized_axes};
}
}  // namespace ngraph

namespace ov {
namespace op {
namespace util {
std::vector<PartialShape> get_tensors_partial_shapes(const TensorVector& tensors) {
    std::vector<PartialShape> shapes;
    shapes.reserve(tensors.size());
    for (const auto& t : tensors) {
        shapes.emplace_back(t.get_shape());
    }
    return shapes;
}
}  // namespace util
}  // namespace op
}  // namespace ov
