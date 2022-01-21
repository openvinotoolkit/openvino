// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/evaluate_helpers.hpp"

namespace ngraph {
AxisSet get_normalized_axes_from_tensor(const HostTensorPtr tensor,
                                        const ngraph::Rank& rank,
                                        const std::string& node_description) {
    const auto axes_vector = host_tensor_2_vector<int64_t>(tensor);
    const auto normalized_axes = ngraph::normalize_axes(node_description, axes_vector, rank);
    return AxisSet{normalized_axes};
}
}  // namespace ngraph
