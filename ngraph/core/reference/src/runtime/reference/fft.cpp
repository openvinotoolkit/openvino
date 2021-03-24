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
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <vector>
#include "ngraph/runtime/reference/fft.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace ngraph::runtime::reference;

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace
            {
                std::vector<int64_t> compute_strides(const std::vector<int64_t>& v)
                {
                    std::vector<int64_t> strides(v.size() + 1);
                    int64_t stride = 1;
                    for (size_t i = 0; i < v.size(); ++i)
                    {
                        strides[i] = stride;
                        stride *= v[i]
                    }
                    strides.back() = stride;
                    return strides;
                }
            }

            void fft(const float* input_data,
                     const Shape& input_data_shape,
                     const int64_t* axes_data,
                     const Shape& axes_data_shape,
                     const int64_t* signal_size_data,
                     const Shape& signal_size_data_shape,
                     float* fft_result,
                     const Shape& output_shape,
                     FFTKind fft_kind)
            {
                using complex_type = std::complex<float>;

                const complex_type* complex_input_data_ptr =
                    reinterpret_cast<const complex_type*>(input_data);
                complex_type* complex_output_ptr = reinterpret_cast<complex_type*>(fft_result);

                size_t complex_data_rank = input_data_shape.size() - 1;

                std::vector<int64_t> reversed_shape(complex_data_rank);
                for (size_t i = 0; i < complex_data_rank; ++i)
                {
                    reversed_shape[i] =
                        static_cast<int64_t>(output_shape[complex_data_rank - i - 1]);
                }

                auto strides = compute_strides(reversed_shape);

                size_t num_of_fft_axes = axes_data_shape[0];

                std::vector<int64_t> sorted_axes(num_of_fft_axes);
                memcpy(sorted_axes.data(), axes_data, num_of_fft_axes * sizeof(int64_t));
                std::sort(sorted_axes.begin(), sorted_axes.end(), std::greater<int64_t>());

                std::vector<int64_t> fft_axes(num_of_fft_axes);
                std::vector<int64_t> fft_lengths(num_of_fft_axes);
                std::vector<int64_t> fft_strides(num_of_fft_axes);

                int64_t fft_size = 1;

                for (size_t i = 0; i < num_of_fft_axes; ++i)
                {
                    int64_t a = sorted_axes[i];
                    int64_t fft_axis = complex_data_rank - 1 - a;
                    fft_axes[i] = fft_axis;
                    fft_lengths[i] = reversed_shape[fft_axis];
                    fft_strides[i] = strides[fft_axis];
                    fft_size *= fft_lengths[i];
                }

                if (fft_size <= 0)
                {
                    return;
                }

                int64_t fft_axes_as_bitset = 0;
                for (int64_t axis : fft_axes)
                {
                    fft_axes_as_bitset |= static_cast<int64_t>(1) << axis;
                }

                int64_t outer_rank = static_cast<int64_t>(complex_data_rank - num_of_fft_axes);
                std::vector<int64_t> outer_axes(outer_rank);
                for (size_t j = 0, i = 0; i < complex_data_rank; ++i)
                {
                    if (fft_axes_as_bitset & (static_cast<int64_t>(1) << i))
                    {
                        outer_axes[j] = i;
                        ++j;
                    }
                }
            }

            void fft_postprocessing(const HostTensorVector& outputs,
                                    const ngraph::element::Type output_type,
                                    const std::vector<float>& fft_result)
            {
                size_t fft_result_size = fft_result.size();

                switch (output_type)
                {
                case element::Type_t::bf16:
                {
                    bfloat16* result_ptr = outputs[0]->get_data_ptr<bfloat16>();
                    for (size_t i = 0; i < fft_result_size; ++i)
                    {
                        result_ptr[i] = bfloat16(fft_result[i]);
                    }
                }
                break;
                case element::Type_t::f16:
                {
                    float16* result_ptr = outputs[0]->get_data_ptr<float16>();
                    for (size_t i = 0; i < fft_result_size; ++i)
                    {
                        result_ptr[i] = float16(fft_result[i]);
                    }
                }
                break;
                case element::Type_t::f32:
                {
                    float* result_ptr = outputs[0]->get_data_ptr<float>();
                    memcpy(result_ptr, fft_result.data(), fft_result_size * sizeof(float));
                }
                break;
                default:;
                }
            }
        }
    }
}