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

#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void evaluate_region_yolo(const T* input,
                                      const Shape& input_shape,
                                      const int coords,
                                      const int classes,
                                      const int regions,
                                      const bool do_softmax,
                                      const std::vector<int64_t>& mask,
                                      const int axis,
                                      const int end_axis,
                                      const std::vector<float>& anchors,
                                      T* output,
                                      const Shape& output_shape)
            {
                
            }

        } // namespace reference

    } //namespace runtime

} // namespace ngraph