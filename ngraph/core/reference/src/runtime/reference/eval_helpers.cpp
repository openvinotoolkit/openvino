// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "ngraph/check.hpp"
#include "ngraph/runtime/reference/eval_helpers.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace eval
    {
        AxisSet extract_reduction_axes(const HostTensorPtr& axes, const char* op_name)
        {
            const auto axes_in_tensor = host_tensor_2_vector<int64_t>(axes);

            const bool negative_axis_received =
                std::any_of(axes_in_tensor.begin(), axes_in_tensor.end(), [](const int64_t axis) {
                    return axis < 0;
                });

            NGRAPH_CHECK(!negative_axis_received,
                         "Negative axis value received in the ",
                         op_name,
                         " evaluation. This case is not supported.");

            return AxisSet(
                std::vector<AxisSet::value_type>(axes_in_tensor.begin(), axes_in_tensor.end()));
        }
    } // namespace eval
} // namespace ngraph
