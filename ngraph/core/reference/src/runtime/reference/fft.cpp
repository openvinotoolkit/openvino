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

#include <iostream>
#include "ngraph/runtime/reference/fft.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>
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
                using complex_type = std::complex<float>;

                std::vector<int64_t> compute_strides(const std::vector<int64_t>& v)
                {
                    std::vector<int64_t> strides(v.size() + 1);
                    int64_t stride = 1;
                    for (size_t i = 0; i < v.size(); ++i)
                    {
                        strides[i] = stride;
                        stride *= v[i];
                    }
                    strides.back() = stride;
                    return strides;
                }

                std::vector<int64_t> reverse_shape(const Shape& shape)
                {
                    size_t complex_data_rank = shape.size() - 1;

                    std::vector<int64_t> reversed_shape(complex_data_rank);
                    for (size_t i = 0; i < complex_data_rank; ++i)
                    {
                        reversed_shape[i] = static_cast<int64_t>(shape[complex_data_rank - i - 1]);
                    }
                    return reversed_shape;
                }

                std::vector<int64_t> canonicalize_axes(const int64_t* axes_data,
                                                       const Shape& axes_data_shape,
                                                       int64_t complex_data_rank)
                {
                    size_t num_of_fft_axes = axes_data_shape[0];

                    std::vector<int64_t> result(axes_data, axes_data + num_of_fft_axes);
                    for (int64_t& axis : result)
                    {
                        if (axis < 0)
                        {
                            axis += complex_data_rank;
                        }
                    }
                    return result;
                }

                std::vector<int64_t> get_axes(const int64_t* axes_data,
                                              const Shape& axes_data_shape,
                                              int64_t complex_data_rank)
                {
                    auto axes = canonicalize_axes(axes_data, axes_data_shape, complex_data_rank);
                    std::sort(axes.begin(), axes.end());
                    return axes;
                }

                void reverse_fft_axes(std::vector<int64_t>& axes, int64_t complex_data_rank)
                {
                    for (int64_t& axis : axes)
                    {
                        axis = complex_data_rank - 1 - axis;
                    }
                }

                std::vector<int64_t> get_lengths(const std::vector<int64_t>& shape,
                                                 const std::vector<int64_t>& axes)
                {
                    std::vector<int64_t> lengths;
                    for (int64_t axis : axes)
                    {
                        lengths.push_back(shape[axis]);
                    }
                    return lengths;
                }

                std::vector<int64_t> get_outer_axes(const std::vector<int64_t>& inner_axes,
                                                    int64_t complex_data_rank)
                {
                    int64_t num_of_inner_axes = static_cast<int64_t>(inner_axes.size());
                    int64_t num_of_outer_axes = complex_data_rank - num_of_inner_axes;

                    std::vector<int64_t> outer_axes(num_of_outer_axes);

                    int64_t fft_axes_as_bitset = 0;
                    for (int64_t axis : inner_axes)
                    {
                        fft_axes_as_bitset |= static_cast<int64_t>(1) << axis;
                    }

                    for (int64_t j = 0, i = 0; i < complex_data_rank; ++i)
                    {
                        if ((fft_axes_as_bitset & (static_cast<int64_t>(1) << i)) == 0)
                        {
                            outer_axes[j] = i;
                            ++j;
                        }
                    }

                    return outer_axes;
                }

                inline bool is_power_of_two(int64_t x) { return (x != 0) && ((x & (x - 1)) == 0); }

                int64_t compute_buffer_size(const std::vector<int64_t>& fft_lengths)
                {
                    int64_t buffer_size = 0;

                    for (int64_t length : fft_lengths)
                    {
                        int64_t current_size = is_power_of_two(length) ? (2 * length) : length;
                        buffer_size = std::max(buffer_size, current_size);
                    }

                    return buffer_size;
                }

                std::vector<int64_t> coords_from_index(int64_t index,
                                                       const std::vector<int64_t>& strides)
                {
                    int64_t num_of_axes = static_cast<int64_t>(strides.size()) - 1;
                    std::vector<int64_t> coords(num_of_axes);
                    int64_t curr = index;
                    for (int64_t j = num_of_axes - 1; j >= 1; --j)
                    {
                        coords[j] = curr / strides[j];
                        curr %= strides[j];
                    }
                    coords[0] = curr;
                    return coords;
                }

                complex_type get_value_from_input(const complex_type* input_data,
                                                  int64_t src_index,
                                                  const std::vector<int64_t>& coords,
                                                  const std::vector<int64_t>& input_fft_lengths,
                                                  const std::vector<int64_t>& input_fft_strides)
                {
                    int64_t offset = 0;
                    int64_t num_of_fft_axes = static_cast<int64_t>(coords.size());
                    for (int64_t i = 0; i < num_of_fft_axes; ++i)
                    {
                        int64_t coord = coords[i];
                        if (coord >= input_fft_lengths[i])
                        {
                            return complex_type{0.0f, 0.0f};
                        }
                        offset += coord * input_fft_strides[i];
                    }

                    return input_data[src_index + offset];
                }

                void copy_data_from_input(complex_type* result,
                                          const complex_type* input_data,
                                          int64_t src_index,
                                          int64_t fft_size,
                                          const std::vector<int64_t>& fft_strides,
                                          const std::vector<int64_t>& input_fft_lengths,
                                          const std::vector<int64_t>& input_fft_strides)
                {
                    for (int64_t idx = 0; idx < fft_size; ++idx)
                    {
                        auto coords = coords_from_index(idx, fft_strides);
                        complex_type value = get_value_from_input(
                            input_data, src_index, coords, input_fft_lengths, input_fft_strides);
                        result[idx] = value;
                    }
                }

                bool blob_is_zero(const complex_type* data, int64_t blob_size)
                {
                    for (int64_t i = 0; i < blob_size; ++i)
                    {
                        if (data[i] != complex_type{0.0f, 0.0f})
                        {
                            return false;
                        }
                    }
                    return true;
                }

                void copy_data_to_output(complex_type* output,
                                         const complex_type* data,
                                         int64_t dst_index,
                                         int64_t fft_size,
                                         const std::vector<int64_t>& fft_strides,
                                         const std::vector<int64_t>& output_fft_strides)
                {
                    int64_t num_of_fft_axes = static_cast<int64_t>(fft_strides.size());

                    for (int64_t idx = 0; idx < fft_size; ++idx)
                    {
                        auto coords = coords_from_index(idx, fft_strides);
                        complex_type value = data[idx];

                        int64_t offset = 0;
                        for (int64_t i = 0; i < num_of_fft_axes; ++i)
                        {
                            int64_t coord = coords[i];
                            offset += coord * output_fft_strides[i];
                        }

                        output[dst_index + offset] = value;
                    }
                }

                static constexpr float pi = 3.141592653589793238462643f;

                complex_type twiddle(int64_t k, int64_t length, FFTKind fft_kind)
                {
                    float angle = -2.0f * pi * static_cast<float>(k) / static_cast<float>(length);
                    complex_type result = std::polar(1.0f, angle);
                    return (fft_kind == FFTKind::Inverse) ? std::conj(result) : result;
                }

                bool gather_to_buffer(const complex_type* data,
                                      int64_t length,
                                      int64_t start,
                                      int64_t stride,
                                      complex_type* buffer)
                {
                    bool input_is_zero = true;
                    for (int64_t k = 0; k < length; ++k)
                    {
                        complex_type value = data[start + k * stride];
                        buffer[k] = value;
                        input_is_zero = input_is_zero && (value == complex_type(0.0f, 0.0f));
                    }
                    return input_is_zero;
                }

                std::vector<complex_type> generate_twiddles(int64_t length, FFTKind fft_kind)
                {
                    std::vector<complex_type> twiddles(length / 2);
                    for (int64_t k = 0; k < length / 2; ++k)
                    {
                        twiddles[k] = twiddle(k, length, fft_kind);
                    }
                    return twiddles;
                }

                void optimized_fft1d(int64_t length,
                                     int64_t fft_offset,
                                     int64_t stride,
                                     complex_type* data,
                                     complex_type* buffer,
                                     FFTKind fft_kind)
                {
                    bool input_is_zero = gather_to_buffer(data, length, fft_offset, stride, buffer);
                    if (input_is_zero)
                    {
                        return;
                    }

                    int64_t in_base = length;
                    int64_t out_base = 0;
                    for (int64_t num_blocks = 1; num_blocks < length; num_blocks *= 2)
                    {
                        std::swap(in_base, out_base);

                        auto twiddles = generate_twiddles(num_blocks * 2, fft_kind);
                        const int64_t block_size = length / num_blocks;
                        const int64_t next_iteration_block_size = block_size / 2;
                        for (int64_t block = 0; block < num_blocks; block++)
                        {
                            const int64_t in_offset = in_base + block * block_size;
                            const int64_t out_offset = out_base + block * next_iteration_block_size;

                            for (int64_t pair = 0; pair < block_size / 2; pair++)
                            {
                                const complex_type even = buffer[in_offset + pair];
                                const complex_type odd = buffer[in_offset + block_size / 2 + pair];
                                const complex_type twiddled_odd = twiddles[block] * odd;
                                buffer[out_offset + pair] = even + twiddled_odd;
                                buffer[out_offset + length / 2 + pair] = even - twiddled_odd;
                            }
                        }
                    }

                    for (int64_t k = 0; k < length; k++)
                    {
                        complex_type value = buffer[out_base + k];
                        if (fft_kind == FFTKind::Inverse)
                        {
                            value /= complex_type(length, 0.0f);
                        }
                        data[fft_offset + k * stride] = value;
                    }
                }

                void naive_fft1d(int64_t length,
                                 int64_t fft_offset,
                                 int64_t stride,
                                 complex_type* data,
                                 complex_type* buffer,
                                 FFTKind fft_kind)
                {
                    bool input_is_zero = gather_to_buffer(data, length, fft_offset, stride, buffer);
                    if (input_is_zero)
                    {
                        return;
                    }
                    for (int64_t k = 0; k < length; ++k)
                    {
                        complex_type value = complex_type(0.0f, 0.0f);
                        for (int64_t n = 0; n < length; ++n)
                        {
                            value += buffer[n] * twiddle(n * k, length, fft_kind);
                        }
                        if (fft_kind == FFTKind::Inverse)
                        {
                            value /= complex_type(length, 0.0f);
                        }
                        data[fft_offset + k * stride] = value;
                    }
                }

                void fft1d(int64_t length,
                           int64_t fft_offset,
                           int64_t stride,
                           complex_type* data,
                           complex_type* buffer,
                           FFTKind fft_kind)
                {
                    if (is_power_of_two(length))
                    {
                        optimized_fft1d(length, fft_offset, stride, data, buffer, fft_kind);
                    }
                    else
                    {
                        naive_fft1d(length, fft_offset, stride, data, buffer, fft_kind);
                    }
                }
            }

            void fft(const float* input_data,
                     const Shape& input_data_shape,
                     const int64_t* axes_data,
                     const Shape& axes_data_shape,
                     float* fft_result,
                     const Shape& output_shape,
                     FFTKind fft_kind)
            {
                const complex_type* complex_input_data_ptr =
                    reinterpret_cast<const complex_type*>(input_data);
                complex_type* complex_output_ptr = reinterpret_cast<complex_type*>(fft_result);

                const int64_t complex_data_rank = static_cast<int64_t>(input_data_shape.size() - 1);
                std::cout << "Input data shape: " << input_data_shape << "\n";
                std::cout << "Axes data shape: " << axes_data_shape << "\n";
                std::cout << "Output shape: " << output_shape << "\n";
                std::cout << "Complex data rank: " << complex_data_rank << "\n";

                const auto reversed_output_shape = reverse_shape(output_shape);
                std::cout << "Reversed output shape: [ ";
                for (auto v : reversed_output_shape)
                {
                    std::cout << v << " ";
                }
                std::cout << "]\n";
                auto fft_axes = get_axes(axes_data, axes_data_shape, complex_data_rank);
                std::cout << "FFT axes: [ ";
                for (auto a : fft_axes)
                {
                    std::cout << a << " ";
                }
                std::cout << "]\n";
                reverse_fft_axes(fft_axes, complex_data_rank);
                std::cout << "Reversed FFT axes: [ ";
                for (auto a : fft_axes)
                {
                    std::cout << a << " ";
                }
                std::cout << "]\n";

                const int64_t fft_rank = fft_axes.size();
                std::cout << "fft_rank: " << fft_rank << "\n";
                const auto fft_lengths = get_lengths(reversed_output_shape, fft_axes);
                std::cout << "fft_lengths: [ ";
                for (auto a : fft_lengths)
                {
                    std::cout << a << " ";
                }
                std::cout << "]\n";
                const auto fft_strides = compute_strides(fft_lengths);
                std::cout << "fft_strides: [ ";
                for (auto a : fft_strides)
                {
                    std::cout << a << " ";
                }
                std::cout << "]\n";
                const int64_t fft_size = fft_strides[fft_rank];
                std::cout << "fft_size: " << fft_size << "\n";

                if (fft_size <= 0)
                {
                    return;
                }

                std::cout << "fft_size is posiztive\n";
                std::vector<complex_type> data(fft_size);

                const auto outer_axes = get_outer_axes(fft_axes, complex_data_rank);

                const int64_t outer_rank = outer_axes.size();
                const auto outer_lengths = get_lengths(reversed_output_shape, outer_axes);
                const auto outer_strides = compute_strides(outer_lengths);
                const int64_t outer_size = outer_strides[outer_rank];

                int64_t buffer_size = compute_buffer_size(fft_lengths);
                std::vector<complex_type> buffer(buffer_size);

                const auto output_strides = compute_strides(reversed_output_shape);
                const auto output_fft_strides = get_lengths(output_strides, fft_axes);
                const auto reversed_input_shape = reverse_shape(input_data_shape);
                const auto input_fft_lengths = get_lengths(reversed_input_shape, fft_axes);
                const auto input_strides = compute_strides(reversed_input_shape);
                const auto input_fft_strides = get_lengths(input_strides, fft_axes);

                for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx)
                {
                    const auto outer_coords = coords_from_index(outer_idx, outer_strides);
                    copy_data_from_input(data.data(),
                                         complex_input_data_ptr,
                                         outer_idx,
                                         fft_size,
                                         fft_strides,
                                         input_fft_lengths,
                                         input_fft_strides);

                    bool input_is_zero = blob_is_zero(data.data(), fft_size);
                    if (!input_is_zero)
                    {
                        for (int64_t axis_idx = 0; axis_idx < fft_rank; ++axis_idx)
                        {
                            int64_t current_fft_stride = fft_strides[axis_idx];
                            int64_t current_fft_length = fft_lengths[axis_idx];

                            int64_t outer_fft_size = 1;
                            for (int64_t i = 0; i < fft_rank; ++i)
                            {
                                if (i == axis_idx)
                                {
                                    continue;
                                }
                                outer_fft_size *= fft_lengths[i];
                            }

                            for (int64_t outer_fft_idx = 0; outer_fft_idx < outer_fft_size;
                                 ++outer_fft_idx)
                            {
                                fft1d(current_fft_length,
                                      outer_fft_idx,
                                      current_fft_stride,
                                      data.data(),
                                      buffer.data(),
                                      fft_kind);
                            }
                        }
                    }

                    copy_data_to_output(complex_output_ptr,
                                        data.data(),
                                        outer_idx,
                                        fft_size,
                                        fft_strides,
                                        output_fft_strides);
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