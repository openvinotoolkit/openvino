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

#include "ngraph/runtime/reference/convolution_backprop_data.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename INPUT,
                      typename FILTER,
                      typename OUTPUT,
                      typename ACCU = typename widen<OUTPUT>::type>
            void group_convolution_backprop_data(const INPUT* in,
                                                 const FILTER* f,
                                                 OUTPUT* out,
                                                 const Shape& in_shape,
                                                 const Shape& filter_shape,
                                                 const Shape& out_shape,
                                                 const Strides& strides,
                                                 const Strides& dilation,
                                                 const CoordinateDiff& pads_begin,
                                                 const CoordinateDiff& pads_end)

            {
                const size_t group_count = filter_shape[filter_group_axis];

                const INPUT* group_batch = in;
                const Shape group_batch_shape = [&]() {
                    Shape new_shape{in_shape};
                    new_shape[in_batch_axis] = 1;
                    new_shape[in_channel_axis] /= group_count;
                    return new_shape;
                }();
                const size_t group_batch_size = shape_size(group_batch_shape);

                const FILTER* group_filter = f;
                const Shape group_filter_shape = [&]() {
                    Shape new_shape{++filter_shape.begin(), filter_shape.end()};
                    return new_shape;
                }();
                const size_t group_filter_size = shape_size(group_filter_shape);

                OUTPUT* group_out = out;
                const Shape group_out_shape = [&]() {
                    Shape new_shape{out_shape};
                    new_shape[out_batch_axis] = 1;
                    new_shape[out_channel_axis] /= group_count;
                    return new_shape;
                }();
                const size_t group_out_size = shape_size(group_out_shape);

                Strides in_dilation(in_shape.size(), 1);
                for (size_t batch_idx = 0; batch_idx < in_shape[in_batch_axis]; ++batch_idx)
                {
                    group_filter = f;
                    for (size_t group_idx = 0; group_idx < group_count; ++group_idx)
                    {
                        runtime::reference::convolution_backprop_in<INPUT, FILTER, OUTPUT, ACCU>(
                            group_batch,
                            group_filter,
                            group_out,
                            group_batch_shape,
                            group_filter_shape,
                            group_out_shape,
                            in_dilation,
                            dilation,
                            pads_begin,
                            pads_end,
                            strides);
                        group_batch += group_batch_size;
                        group_filter += group_filter_size;
                        group_out += group_out_size;
                    }
                }
            }
        } // namespace reference

    } // namespace runtime

} // namespace ngraph