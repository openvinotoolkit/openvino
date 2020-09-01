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

#pragma once

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/roi_align.hpp" // for ROIAlign:PoolingMode
namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void roi_align(const std::vector<const char*>& args,
                           char* out,
                           const std::vector<Shape> arg_shapes,
                           const Shape& out_shape,
                           const std::vector<size_t> arg_elem_sizes,
                           const int pooled_width,
                           const int pooled_height,
                           const int sampling_ratio,
                           const float spatial_scale,
                           const ngraph::op::v3::ROIAlign::PoolingMode pooling_mode);
        }
    } // namespace runtime
} // namespace ngraph