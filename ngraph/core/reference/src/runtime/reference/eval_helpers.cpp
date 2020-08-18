//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

namespace ngraph
{
    namespace eval
    {
        AxisSet extract_reduction_axes(const HostTensorPtr& axes, const char* op_name)
        {
            const auto axes_count = axes->get_element_count();
            const auto axes_buffer = axes->get_data_ptr<int64_t>();

            const bool negative_axis_received = std::any_of(
                axes_buffer, axes_buffer + axes_count, [](const int64_t axis) { return axis < 0; });

            NGRAPH_CHECK(!negative_axis_received,
                         "Negative axis value received in the ",
                         op_name,
                         " evaluation. This case is not supported.");

            return AxisSet(std::vector<AxisSet::value_type>(axes_buffer, axes_buffer + axes_count));
        }
    }
}
