// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "ngraph/check.hpp"
#include "ngraph/runtime/reference/eval_helpers.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

namespace ngraph
{
    namespace eval
    {
        AxisSet extract_reduction_axes(const HostTensorPtr& data,
                                       const HostTensorPtr& axes,
                                       const char* op_name)
        {
            const auto axes_in_tensor = host_tensor_2_vector<int64_t>(axes);
            const auto data_rank = data->get_partial_shape().rank();
            auto norm_axes = normalize_axes(op_name, axes_in_tensor, data_rank);
            return AxisSet(std::vector<AxisSet::value_type>(norm_axes.begin(), norm_axes.end()));
        }
    } // namespace eval
} // namespace ngraph
