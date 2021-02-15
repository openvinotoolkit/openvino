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

#pragma once

#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            struct DeformableConvolutionParams
            {
                std::vector<int> strides;
                std::vector<int> dilation;
                std::vector<int> pads_begin;
                std::vector<int> pads_end;
                int64_t groups;
                int64_t deformable_groups;

                DeformableConvolutionParams(const Strides& strides_,
                                    const Strides& dilation_,
                                    const CoordinateDiff& pads_begin_,
                                    const CoordinateDiff& pads_end_,
                                    const int64_t groups_,
                                    const int64_t deformable_groups_)
                    : strides{strides_.begin(), strides_.end()}
                    , dilation{dilation_.begin(), dilation_.end()}
                    , pads_begin{pads_begin_.begin(), pads_begin_.end()}
                    , pads_end{pads_end_.begin(), pads_end_.end()}
                    , groups{groups_}
                    , deformable_groups{deformable_groups_} {};
            };

            template <typename T>
            void deformable_convolution(const T* in,
                                        const T* f,
                                        T* out,
                                        const Shape& in_shape,
                                        const Shape& f_shape,
                                        const Shape& out_shape,
                                        const Strides& strides,
                                        const Strides& dilation,
                                        const CoordinateDiff& pads_begin,
                                        const CoordinateDiff& pads_end,
                                        const int64_t groups,
                                        const int64_t deformable_groups)

            {
                // this implementation supports 2D convolutions
                NGRAPH_CHECK(in_shape.size() == 4,
                             "Unsupported input rank: ",
                             in_shape);

                NGRAPH_CHECK(f_shape.size() == 4,
                             "Unsupported kernel rank: ",
                             f_shape);

                // here we are converting all param types to int's to avoid arithmetic issues
                // (e.g signed + unsigned) in indexes calculation later
                DeformableConvolutionParams params{strides, dilation, pads_begin, pads_end, groups, deformable_groups};

                const T* group_batch = in;
                const Shape group_batch_shape = [&]() {
                    Shape new_shape{in_shape};
                    new_shape[out_batch_axis] = 1;
                    new_shape[out_channel_axis] /= groups;
                    return new_shape;
                }();
                const size_t group_batch_size = shape_size(group_batch_shape);

                const T* group_filter = f;
                const Shape group_f_shape = [&]() {
                    Shape new_shape{++f_shape.begin(), f_shape.end()};
                    return new_shape;
                }();
                const size_t group_filter_size = shape_size(group_f_shape);

                T* group_out = out;
                const Shape group_out_shape = [&]() {
                    Shape new_shape{out_shape};
                    new_shape[out_batch_axis] = 1;
                    new_shape[out_channel_axis] /= groups;
                    return new_shape;
                }();
                const size_t group_out_size = shape_size(group_out_shape);

                for (size_t batch_idx = 0; batch_idx < in_shape[in_batch_axis]; ++batch_idx)
                {
                    group_filter = f;
                    for (size_t group_idx = 0; group_idx < groups; ++group_idx)
                    {
                        runtime::reference::convolution(group_batch,
                                                        group_filter,
                                                        group_out,
                                                        group_batch_shape,
                                                        group_f_shape,
                                                        group_out_shape,
                                                        strides,
                                                        dilation,
                                                        pads_begin,
                                                        pads_end);
                        group_batch += group_batch_size;
                        group_filter += group_filter_size;
                        group_out += group_out_size;
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph