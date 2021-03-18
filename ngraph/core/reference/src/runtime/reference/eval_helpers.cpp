//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
