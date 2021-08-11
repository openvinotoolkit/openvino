// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/evaluate_helpers.hpp"

namespace ov
{
    AxisSet get_normalized_axes_from_tensor(const HostTensorPtr tensor,
                                            const ov::Rank& rank,
                                            const std::string& node_description)
    {
        const auto axes_vector = host_tensor_2_vector<int64_t>(tensor);
        const auto normalized_axes = ov::normalize_axes(node_description, axes_vector, rank);
        return AxisSet{normalized_axes};
    }
} // namespace ov
