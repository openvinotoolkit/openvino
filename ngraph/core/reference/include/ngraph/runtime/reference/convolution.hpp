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

#include <cassert>
#include <cfenv>
#include <cmath>
#include <functional>
#include <numeric>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/helpers.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/split.hpp"
#include "ngraph/util.hpp"

// can't be removed currently due to arm-plugin dependency
#include "ngraph/runtime/reference/convolution_backprop_data.hpp"
namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace
            {
                constexpr size_t in_batch_axis = 0;
                constexpr size_t in_channel_axis = 1;
                constexpr size_t filter_out_ch_axis = 0;
                constexpr size_t filter_in_ch_axis = 1;
                constexpr size_t out_batch_axis = 0;
                constexpr size_t out_channel_axis = 1;
                constexpr size_t spatial_axis = 2;

                template <typename T>
                struct Tensor
                {
                    T buffer;
                    const Shape shape;
                    Tensor(T buf, Shape s)
                        : buffer{buf}
                        , shape{s} {};
                };

                enum class ConvolutionType
                {
                    Conv1D,
                    Conv2D,
                    Conv3D
                };

                struct ConvolutionParams
                {
                    std::vector<int> strides;
                    std::vector<int> dilation;
                    std::vector<int> pads_begin;
                    std::vector<int> pads_end;

                    ConvolutionParams(const Strides& strides_,
                                      const Strides& dilation_,
                                      const CoordinateDiff& pads_begin_,
                                      const CoordinateDiff& pads_end_)
                        : strides{strides_.begin(), strides_.end()}
                        , dilation{dilation_.begin(), dilation_.end()}
                        , pads_begin{pads_begin_.begin(), pads_begin_.end()}
                        , pads_end{pads_end_.begin(), pads_end_.end()} {};
                };

                size_t compute_size(ngraph::Shape shape, size_t start_axis)
                {
                    size_t size = 1;
                    for (size_t i = start_axis; i < shape.size(); i++)
                    {
                        size *= shape[i];
                    }
                    return size;
                }

                template <typename INPUT, typename FILTER, typename OUTPUT, typename ACCU>
                void convolve_1D_channels(const ConvolutionParams& p,
                                          const Tensor<const INPUT*>& in,
                                          const Tensor<const FILTER*>& f,
                                          Tensor<OUTPUT*>& out)
                {
                    const int input_spatial_size = compute_size(in.shape, spatial_axis);
                    const int filter_spatial_size = compute_size(f.shape, spatial_axis);
                    const int input_size_x = in.shape[spatial_axis];
                    const int filter_size_x = f.shape[spatial_axis];
                    const int dilated_filter_size_x =
                        filter_size_x + (filter_size_x - 1) * (p.dilation[0] - 1);

                    for (int i_x = -p.pads_begin[0];
                         i_x <= (p.pads_end[0] + input_size_x - dilated_filter_size_x);
                         i_x += p.strides[0])
                    {
                        ACCU sum = 0;
                        Tensor<const INPUT*> ch_input = in;
                        Tensor<const INPUT*> ch_filter = f;
                        for (size_t ch_idx = 0; ch_idx < f.shape[filter_in_ch_axis]; ++ch_idx)
                        {
                            for (int f_x = 0; f_x < filter_size_x; ++f_x)
                            {
                                int rel_i_x = i_x + (f_x * p.dilation[0]);
                                bool padding = (rel_i_x < 0) || (rel_i_x >= input_size_x);
                                if (padding)
                                    continue;

                                int f_buf_idx = f_x;
                                int i_buf_idx = rel_i_x;

                                sum += static_cast<ACCU>(ch_input.buffer[i_buf_idx]) *
                                       static_cast<ACCU>(ch_filter.buffer[f_buf_idx]);
                            }
                            ch_input.buffer += input_spatial_size;
                            ch_filter.buffer += filter_spatial_size;
                        }
                        *out.buffer = sum;
                        ++out.buffer;
                    }
                }

                template <typename INPUT, typename FILTER, typename OUTPUT, typename ACCU>
                void convolve_2D_channels(const ConvolutionParams& p,
                                          const Tensor<const INPUT*>& in,
                                          const Tensor<const FILTER*>& f,
                                          Tensor<OUTPUT*>& out)
                {
                    const int input_spatial_size = compute_size(in.shape, spatial_axis);
                    const int filter_spatial_size = compute_size(f.shape, spatial_axis);
                    const int input_size_y = in.shape[spatial_axis];
                    const int input_size_x = in.shape[spatial_axis + 1];
                    const int filter_size_y = f.shape[spatial_axis];
                    const int filter_size_x = f.shape[spatial_axis + 1];
                    const int dilated_filter_size_y =
                        filter_size_y + (filter_size_y - 1) * (p.dilation[0] - 1);
                    const int dilated_filter_size_x =
                        filter_size_x + (filter_size_x - 1) * (p.dilation[1] - 1);

                    for (int i_y = -p.pads_begin[0];
                         i_y <= (p.pads_end[0] + input_size_y - dilated_filter_size_y);
                         i_y += p.strides[0])
                    {
                        for (int i_x = -p.pads_begin[1];
                             i_x <= (p.pads_end[1] + input_size_x - dilated_filter_size_x);
                             i_x += p.strides[1])
                        {
                            ACCU sum = 0;
                            Tensor<const INPUT*> ch_input = in;
                            Tensor<const INPUT*> ch_filter = f;
                            for (size_t ch_idx = 0; ch_idx < f.shape[filter_in_ch_axis]; ++ch_idx)
                            {
                                for (int f_y = 0; f_y < filter_size_y; ++f_y)
                                {
                                    for (int f_x = 0; f_x < filter_size_x; ++f_x)
                                    {
                                        int rel_i_y = i_y + (f_y * p.dilation[0]);
                                        int rel_i_x = i_x + (f_x * p.dilation[1]);

                                        bool padding = (rel_i_y < 0) || (rel_i_x < 0) ||
                                                       (rel_i_y >= input_size_y) ||
                                                       (rel_i_x >= input_size_x);
                                        if (padding)
                                            continue;

                                        int f_buf_idx = (f_y * filter_size_x) + f_x;
                                        int i_buf_idx = (rel_i_y * input_size_x) + rel_i_x;
                                        sum += static_cast<ACCU>(ch_input.buffer[i_buf_idx]) *
                                               static_cast<ACCU>(ch_filter.buffer[f_buf_idx]);
                                    }
                                }
                                ch_input.buffer += input_spatial_size;
                                ch_filter.buffer += filter_spatial_size;
                            }
                            *out.buffer = sum;
                            ++out.buffer;
                        }
                    }
                }

                template <typename INPUT, typename FILTER, typename OUTPUT, typename ACCU>
                void convolve_3D_channels(const ConvolutionParams& p,
                                          const Tensor<const INPUT*>& in,
                                          const Tensor<const FILTER*>& f,
                                          Tensor<OUTPUT*>& out)
                {
                    const int input_spatial_size = compute_size(in.shape, spatial_axis);
                    const int filter_spatial_size = compute_size(f.shape, spatial_axis);
                    const int input_size_z = in.shape[spatial_axis];
                    const int input_size_y = in.shape[spatial_axis + 1];
                    const int input_size_x = in.shape[spatial_axis + 2];
                    const int filter_size_z = f.shape[spatial_axis];
                    const int filter_size_y = f.shape[spatial_axis + 1];
                    const int filter_size_x = f.shape[spatial_axis + 2];
                    const int dilated_filter_size_z =
                        filter_size_z + (filter_size_z - 1) * (p.dilation[0] - 1);
                    const int dilated_filter_size_y =
                        filter_size_y + (filter_size_y - 1) * (p.dilation[1] - 1);
                    const int dilated_filter_size_x =
                        filter_size_x + (filter_size_x - 1) * (p.dilation[2] - 1);

                    for (int i_z = -p.pads_begin[0];
                         i_z <= (p.pads_end[0] + input_size_z - dilated_filter_size_z);
                         i_z += p.strides[0])
                    {
                        for (int i_y = -p.pads_begin[1];
                             i_y <= (p.pads_end[1] + input_size_y - dilated_filter_size_y);
                             i_y += p.strides[1])
                        {
                            for (int i_x = -p.pads_begin[2];
                                 i_x <= (p.pads_end[2] + input_size_x - dilated_filter_size_x);
                                 i_x += p.strides[2])
                            {
                                ACCU sum = 0;
                                Tensor<const INPUT*> ch_input = in;
                                Tensor<const INPUT*> ch_filter = f;
                                for (size_t ch_idx = 0; ch_idx < f.shape[filter_in_ch_axis];
                                     ++ch_idx)
                                {
                                    for (int f_z = 0; f_z < filter_size_z; ++f_z)
                                    {
                                        for (int f_y = 0; f_y < filter_size_y; ++f_y)
                                        {
                                            for (int f_x = 0; f_x < filter_size_x; ++f_x)
                                            {
                                                int rel_i_z = i_z + (f_z * p.dilation[0]);
                                                int rel_i_y = i_y + (f_y * p.dilation[1]);
                                                int rel_i_x = i_x + (f_x * p.dilation[2]);

                                                bool padding = (rel_i_z < 0) || (rel_i_y < 0) ||
                                                               (rel_i_x < 0) ||
                                                               (rel_i_z >= input_size_z) ||
                                                               (rel_i_y >= input_size_y) ||
                                                               (rel_i_x >= input_size_x);
                                                if (padding)
                                                    continue;

                                                int f_buf_idx =
                                                    (f_z * filter_size_y * filter_size_x) +
                                                    (f_y * filter_size_x) + f_x;
                                                int i_buf_idx =
                                                    (rel_i_z * input_size_y * input_size_x) +
                                                    (rel_i_y * input_size_x) + rel_i_x;
                                                sum +=
                                                    static_cast<ACCU>(ch_input.buffer[i_buf_idx]) *
                                                    static_cast<ACCU>(ch_filter.buffer[f_buf_idx]);
                                            }
                                        }
                                    }
                                    ch_input.buffer += input_spatial_size;
                                    ch_filter.buffer += filter_spatial_size;
                                }
                                *out.buffer = sum;
                                ++out.buffer;
                            }
                        }
                    }
                }

                template <typename INPUT, typename FILTER, typename OUTPUT, typename ACCU>
                void convolve_channels(const ConvolutionType& type,
                                       const ConvolutionParams& p,
                                       const Tensor<const INPUT*>& in,
                                       const Tensor<const FILTER*>& f,
                                       Tensor<OUTPUT*>& out)
                {
                    switch (type)
                    {
                    case ConvolutionType::Conv1D:
                        convolve_1D_channels<INPUT, FILTER, OUTPUT, ACCU>(p, in, f, out);
                        break;
                    case ConvolutionType::Conv2D:
                        convolve_2D_channels<INPUT, FILTER, OUTPUT, ACCU>(p, in, f, out);
                        break;
                    case ConvolutionType::Conv3D:
                        convolve_3D_channels<INPUT, FILTER, OUTPUT, ACCU>(p, in, f, out);
                        break;
                    }
                }

                template <typename INPUT, typename FILTER, typename OUTPUT, typename ACCU>
                void conv_impl(const ConvolutionType& type,
                               const ConvolutionParams& params,
                               const Tensor<const INPUT*> in,
                               const Tensor<const FILTER*> f,
                               Tensor<OUTPUT*> out)
                {
                    const size_t batch_size = compute_size(in.shape, in_channel_axis);
                    const size_t filter_size = compute_size(f.shape, filter_in_ch_axis);

                    Tensor<const INPUT*> batch = in;
                    for (size_t batch_idx = 0; batch_idx < in.shape[in_batch_axis]; ++batch_idx)
                    {
                        Tensor<const FILTER*> filter = f;
                        for (size_t f_idx = 0; f_idx < f.shape[filter_out_ch_axis]; ++f_idx)
                        {
                            convolve_channels<INPUT, FILTER, OUTPUT, ACCU>(
                                type, params, batch, filter, out);
                            filter.buffer += filter_size;
                        }
                        batch.buffer += batch_size;
                    }
                }
            }

            template <typename INPUT,
                      typename FILTER,
                      typename OUTPUT,
                      typename ACCU = typename widen<OUTPUT>::type>
            void convolution(const INPUT* in,
                             const FILTER* f,
                             OUTPUT* out,
                             const Shape& in_shape,
                             const Shape& filter_shape,
                             const Shape& out_shape,
                             const Strides& strides,
                             const Strides& dilation,
                             const CoordinateDiff& pads_begin,
                             const CoordinateDiff& pads_end,
                             const Strides&)

            {
                const ConvolutionType type = [&]() {
                    switch (filter_shape.size())
                    {
                    case 3: return ConvolutionType::Conv1D;
                    case 4: return ConvolutionType::Conv2D;
                    case 5: return ConvolutionType::Conv3D;
                    default: NGRAPH_UNREACHABLE("Unsupported kernel rank: ", filter_shape);
                    }
                }();

                auto old_mode = std::fegetround();
                std::fesetround(FE_TONEAREST);

                // here we are converting all param types to int's to avoid arithmetic issues
                // (e.g signed + unsigned) in indexes calculation later
                const ConvolutionParams params{strides, dilation, pads_begin, pads_end};

                const Tensor<const INPUT*> input{in, in_shape};
                const Tensor<const FILTER*> filter{f, filter_shape};
                const Tensor<OUTPUT*> output{out, out_shape};

                conv_impl<INPUT, FILTER, OUTPUT, ACCU>(type, params, input, filter, output);

                std::fesetround(old_mode);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
