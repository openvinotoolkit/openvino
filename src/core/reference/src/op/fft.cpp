//*****************************************************************************
// Copyright 2017-2022 Intel Corporation
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

#include "openvino/reference/fft.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include "openvino/reference/utils/fft_common.hpp"

namespace ov {
namespace reference {
// FFT operation supports for negative axes to transform. More precisely, according to
// the FFT operation specification, axes should be integers from -(r - 1) to (r - 2)
// inclusively, where r = rank(data). A negative axis 'a' is interpreted as an axis
// 'r - 1 + a'. The reason is the following: real input tensor of the shape
// [n_0, ..., n_{r - 1}, 2] is interpreted as a complex tensor with the shape
// [n_0, ..., n_{r - 1}]. To simplify calculations, we need to convert negative axes to
// positive axes using the formula 'r - 1 + a'.
std::vector<int64_t> canonicalize_axes(const int64_t* axes_data,
                                       const Shape& axes_data_shape,
                                       int64_t complex_data_rank) {
    size_t num_of_fft_axes = axes_data_shape[0];

    std::vector<int64_t> result(axes_data, axes_data + num_of_fft_axes);
    for (int64_t& axis : result) {
        if (axis < 0) {
            axis += complex_data_rank;
        }
    }
    return result;
}

namespace {
using complex_type = std::complex<float>;

// This function gets FFT axes from axes_data
std::vector<int64_t> get_axes(const int64_t* axes_data, const Shape& axes_data_shape, int64_t complex_data_rank) {
    auto axes = canonicalize_axes(axes_data, axes_data_shape, complex_data_rank);
    std::sort(axes.begin(), axes.end(), std::greater<int64_t>{});
    return axes;
}

// Helper function to get only length with respect to given axes.
std::vector<int64_t> get_lengths(const std::vector<int64_t>& shape, const std::vector<int64_t>& axes) {
    std::vector<int64_t> lengths;
    for (int64_t axis : axes) {
        lengths.push_back(shape[axis]);
    }
    return lengths;
}

// This function calculates 'outer axes', that is axes that are
// not transformed by FFT.
std::vector<int64_t> get_outer_axes(const std::vector<int64_t>& inner_axes, int64_t complex_data_rank) {
    int64_t num_of_inner_axes = static_cast<int64_t>(inner_axes.size());
    int64_t num_of_outer_axes = complex_data_rank - num_of_inner_axes;

    std::vector<int64_t> outer_axes(num_of_outer_axes);

    int64_t fft_axes_as_bitset = 0;
    for (int64_t axis : inner_axes) {
        assert(axis < 64);
        fft_axes_as_bitset |= static_cast<int64_t>(1) << axis;
    }

    for (int64_t j = 0, i = 0; i < complex_data_rank; ++i) {
        if ((fft_axes_as_bitset & (static_cast<int64_t>(1) << i)) == 0) {
            outer_axes[j] = i;
            ++j;
        }
    }

    return outer_axes;
}

inline bool is_power_of_two(int64_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

// This function calculates internal FFT buffer size using lengths of FFT axes.
int64_t compute_buffer_size(const std::vector<int64_t>& fft_lengths) {
    int64_t buffer_size = 0;

    for (int64_t length : fft_lengths) {
        int64_t current_size = is_power_of_two(length) ? (2 * length) : length;
        buffer_size = std::max(buffer_size, current_size);
    }

    return buffer_size;
}

// This function gets a complex value from given coords of this value
complex_type get_value_from_input(const complex_type* input_data,
                                  int64_t src_index,
                                  const std::vector<int64_t>& coords,
                                  const std::vector<int64_t>& input_fft_lengths,
                                  const std::vector<int64_t>& input_fft_strides) {
    int64_t offset = 0;
    int64_t num_of_fft_axes = static_cast<int64_t>(coords.size());
    for (int64_t i = 0; i < num_of_fft_axes; ++i) {
        int64_t coord = coords[i];
        if (coord >= input_fft_lengths[i]) {
            return complex_type{0.0f, 0.0f};
        }
        offset += coord * input_fft_strides[i];
    }

    return input_data[src_index + offset];
}

// Copying input data to the given memory domain.
void copy_data_from_input(complex_type* result,
                          const complex_type* input_data,
                          int64_t src_index,
                          int64_t fft_size,
                          const std::vector<int64_t>& fft_strides,
                          const std::vector<int64_t>& input_fft_lengths,
                          const std::vector<int64_t>& input_fft_strides) {
    for (int64_t idx = 0; idx < fft_size; ++idx) {
        auto coords = fft_common::coords_from_index(idx, fft_strides);
        complex_type value = get_value_from_input(input_data, src_index, coords, input_fft_lengths, input_fft_strides);
        result[idx] = value;
    }
}

// This function checks whether data of given complex blob are only zeros.
bool blob_is_zero(const complex_type* data, int64_t blob_size) {
    for (int64_t i = 0; i < blob_size; ++i) {
        if (data[i] != complex_type{0.0f, 0.0f}) {
            return false;
        }
    }
    return true;
}

// Copying calculated data to the given memory domain.
void copy_data_to_output(complex_type* output,
                         const complex_type* data,
                         int64_t dst_index,
                         int64_t fft_size,
                         const std::vector<int64_t>& fft_strides,
                         const std::vector<int64_t>& output_fft_strides) {
    for (int64_t idx = 0; idx < fft_size; ++idx) {
        auto coords = fft_common::coords_from_index(idx, fft_strides);
        complex_type value = data[idx];
        int64_t offset = fft_common::offset_from_coords_and_strides(coords, output_fft_strides);

        output[dst_index + offset] = value;
    }
}

static constexpr float pi = 3.141592653589793238462643f;

// This function calculates e^{-2i\pi k / length} for the forward FFT, and
// e^{2i\pi k / length} otherwise. Here 'i' is an imaginary unit.
complex_type twiddle(int64_t k, int64_t length, FFTKind fft_kind) {
    float angle = -2.0f * pi * static_cast<float>(k) / static_cast<float>(length);
    complex_type result = std::exp(complex_type(0.0f, angle));
    return (fft_kind == FFTKind::Inverse) ? std::conj(result) : result;
}

// This function gathers data from the input of 1D FFT to the contiguous buffer
void gather_to_buffer(const complex_type* data, int64_t length, int64_t start, int64_t stride, complex_type* buffer) {
    for (int64_t k = 0; k < length; ++k) {
        complex_type value = data[start + k * stride];
        buffer[k] = value;
    }
}

std::vector<complex_type> generate_twiddles(int64_t length, FFTKind fft_kind) {
    std::vector<complex_type> twiddles(length / 2);
    for (int64_t k = 0; k < length / 2; ++k) {
        twiddles[k] = twiddle(k, length, fft_kind);
    }
    return twiddles;
}

// Non-recursive implementation of the Cooley-Tukey radix-2 decimation in
// time. Performs 1D FFT transform for the lengths, which are powers of 2.
// Runs in O(length * log(length)) time. Uses the same parameters as the naive
// implementation above, except that the preallocated buffer must be at least
// twice as big as the length of the transform, because the buffer is used to
// hold both input and output values for each stage of the transform.
void optimized_fft1d(int64_t length,
                     int64_t fft_offset,
                     int64_t stride,
                     complex_type* data,
                     complex_type* buffer,
                     FFTKind fft_kind) {
    gather_to_buffer(data, length, fft_offset, stride, buffer);
    if (blob_is_zero(buffer, length)) {
        return;
    }

    int64_t in_base = length;
    int64_t out_base = 0;
    for (int64_t num_blocks = 1; num_blocks < length; num_blocks *= 2) {
        std::swap(in_base, out_base);

        auto twiddles = generate_twiddles(num_blocks * 2, fft_kind);
        const int64_t block_size = length / num_blocks;
        const int64_t next_iteration_block_size = block_size / 2;
        for (int64_t block = 0; block < num_blocks; block++) {
            const int64_t in_offset = in_base + block * block_size;
            const int64_t out_offset = out_base + block * next_iteration_block_size;

            for (int64_t pair = 0; pair < block_size / 2; pair++) {
                const complex_type even = buffer[in_offset + pair];
                const complex_type odd = buffer[in_offset + block_size / 2 + pair];
                const complex_type twiddled_odd = twiddles[block] * odd;
                buffer[out_offset + pair] = even + twiddled_odd;
                buffer[out_offset + length / 2 + pair] = even - twiddled_odd;
            }
        }
    }

    for (int64_t k = 0; k < length; k++) {
        complex_type value = buffer[out_base + k];
        if (fft_kind == FFTKind::Inverse) {
            value /= complex_type(static_cast<float>(length), 0.0f);
        }
        data[fft_offset + k * stride] = value;
    }
}

// Naive implementation of 1D FFT
void naive_fft1d(int64_t length,
                 int64_t fft_offset,
                 int64_t stride,
                 complex_type* data,
                 complex_type* buffer,
                 FFTKind fft_kind) {
    gather_to_buffer(data, length, fft_offset, stride, buffer);
    if (blob_is_zero(buffer, length)) {
        return;
    }

    for (int64_t k = 0; k < length; ++k) {
        complex_type value = complex_type(0.0f, 0.0f);
        for (int64_t n = 0; n < length; ++n) {
            value += buffer[n] * twiddle(n * k, length, fft_kind);
        }
        if (fft_kind == FFTKind::Inverse) {
            value /= complex_type(static_cast<float>(length), 0.0f);
        }
        data[fft_offset + k * stride] = value;
    }
}

void fft1d(int64_t length,
           int64_t fft_offset,
           int64_t stride,
           complex_type* data,
           complex_type* buffer,
           FFTKind fft_kind) {
    if (is_power_of_two(length)) {
        optimized_fft1d(length, fft_offset, stride, data, buffer, fft_kind);
    } else {
        naive_fft1d(length, fft_offset, stride, data, buffer, fft_kind);
    }
}

struct InfoForFFTCalculation {
    std::vector<int64_t> fft_axes;
    std::vector<int64_t> fft_lengths;
    std::vector<int64_t> fft_strides;
    std::vector<int64_t> outer_strides;
    std::vector<int64_t> output_fft_strides;
    std::vector<int64_t> output_outer_strides;
    std::vector<int64_t> input_fft_lengths;
    std::vector<int64_t> input_fft_strides;
    std::vector<int64_t> input_outer_strides;
    int64_t fft_rank;
    int64_t fft_size;
    int64_t outer_size;
    int64_t buffer_size;
};

// This function builds information needed to calculate FFT.
InfoForFFTCalculation get_info_for_calculation(const Shape& input_data_shape,
                                               const int64_t* axes_data,
                                               const Shape& axes_data_shape,
                                               const Shape& output_shape) {
    InfoForFFTCalculation result;

    const int64_t complex_data_rank = static_cast<int64_t>(input_data_shape.size() - 1);

    const auto reversed_output_shape = fft_common::reverse_shape_of_emulated_complex_tensor(output_shape);
    auto& fft_axes = result.fft_axes;
    fft_axes = get_axes(axes_data, axes_data_shape, complex_data_rank);
    fft_axes = fft_common::reverse_fft_axes(fft_axes, complex_data_rank);

    const int64_t fft_rank = fft_axes.size();
    const auto fft_lengths = get_lengths(reversed_output_shape, fft_axes);
    const auto fft_strides = fft_common::compute_strides(fft_lengths);
    const int64_t fft_size = fft_strides[fft_rank];

    const auto outer_axes = get_outer_axes(fft_axes, complex_data_rank);
    const int64_t outer_rank = outer_axes.size();
    const auto outer_lengths = get_lengths(reversed_output_shape, outer_axes);
    const auto outer_strides = fft_common::compute_strides(outer_lengths);
    const int64_t outer_size = outer_strides[outer_rank];

    const auto output_strides = fft_common::compute_strides(reversed_output_shape);
    const auto reversed_input_shape = fft_common::reverse_shape_of_emulated_complex_tensor(input_data_shape);
    const auto input_strides = fft_common::compute_strides(reversed_input_shape);

    result.fft_lengths = fft_lengths;
    result.fft_strides = fft_strides;
    result.outer_strides = outer_strides;
    result.output_fft_strides = get_lengths(output_strides, fft_axes);
    result.output_outer_strides = get_lengths(output_strides, outer_axes);
    result.input_fft_lengths = get_lengths(reversed_input_shape, fft_axes);
    result.input_fft_strides = get_lengths(input_strides, fft_axes);
    result.input_outer_strides = get_lengths(input_strides, outer_axes);
    result.fft_rank = fft_rank;
    result.fft_size = fft_size;
    result.outer_size = outer_size;
    result.buffer_size = compute_buffer_size(fft_lengths);

    return result;
}

std::vector<int64_t> lengths_except_given_axis(const std::vector<int64_t>& v, int64_t axis) {
    auto result = v;
    if (axis >= 0 && axis < static_cast<int64_t>(v.size()))
        result.erase(result.begin() + axis);
    return result;
}
}  // namespace

// Calculation of FFT
void fft(const float* input_data,
         const Shape& input_data_shape,
         const int64_t* axes_data,
         const Shape& axes_data_shape,
         float* fft_result,
         const Shape& output_shape,
         FFTKind fft_kind) {
    const complex_type* complex_input_data_ptr = reinterpret_cast<const complex_type*>(input_data);
    complex_type* complex_output_ptr = reinterpret_cast<complex_type*>(fft_result);

    const auto info = get_info_for_calculation(input_data_shape, axes_data, axes_data_shape, output_shape);

    const auto& fft_axes = info.fft_axes;
    const int64_t fft_rank = info.fft_rank;
    const auto& fft_lengths = info.fft_lengths;
    const auto& fft_strides = info.fft_strides;
    const int64_t fft_size = info.fft_size;

    if (fft_size <= 0) {
        return;
    }

    std::vector<complex_type> data(fft_size);
    std::vector<complex_type> buffer(info.buffer_size);

    const auto& output_fft_strides = info.output_fft_strides;
    const auto& outer_strides = info.outer_strides;
    const int64_t outer_size = info.outer_size;

    const auto& output_outer_strides = info.output_outer_strides;
    const auto& input_fft_lengths = info.input_fft_lengths;
    const auto& input_fft_strides = info.input_fft_strides;
    const auto& input_outer_strides = info.input_outer_strides;

    // Loop along with 'outer' dimensions, that is along with
    // not transformed dimensions.
    for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
        const auto outer_coords = fft_common::coords_from_index(outer_idx, outer_strides);
        int64_t outer_input_offset = fft_common::offset_from_coords_and_strides(outer_coords, input_outer_strides);

        // Copying current data to transform
        copy_data_from_input(data.data(),
                             complex_input_data_ptr,
                             outer_input_offset,
                             fft_size,
                             fft_strides,
                             input_fft_lengths,
                             input_fft_strides);

        if (!blob_is_zero(data.data(), fft_size)) {
            // The loop along with all transformed axes.
            for (int64_t axis_idx = 0; axis_idx < fft_rank; ++axis_idx) {
                int64_t current_fft_stride = fft_strides[axis_idx];
                int64_t current_fft_length = fft_lengths[axis_idx];
                auto outer_fft_lengths = lengths_except_given_axis(fft_lengths, axis_idx);
                auto outer_fft_axes = lengths_except_given_axis(fft_axes, axis_idx);
                int64_t outer_fft_size = fft_size / current_fft_length;

                auto outer_fft_strides = fft_common::compute_strides(outer_fft_lengths);
                auto fft_strides_for_outer_fft_axes = lengths_except_given_axis(fft_strides, axis_idx);

                // Loop along with all FFT axes, except the current one.
                for (int64_t outer_fft_idx = 0; outer_fft_idx < outer_fft_size; ++outer_fft_idx) {
                    const auto outer_fft_coords = fft_common::coords_from_index(outer_fft_idx, outer_fft_strides);
                    int64_t outer_fft_offset =
                        fft_common::offset_from_coords_and_strides(outer_fft_coords, fft_strides_for_outer_fft_axes);
                    // Calculation of 1D FFT
                    fft1d(current_fft_length,
                          outer_fft_offset,
                          current_fft_stride,
                          data.data(),
                          buffer.data(),
                          fft_kind);
                }
            }
        }

        // Copying current calculated data to the output blob.
        int64_t outer_output_offset = fft_common::offset_from_coords_and_strides(outer_coords, output_outer_strides);
        copy_data_to_output(complex_output_ptr,
                            data.data(),
                            outer_output_offset,
                            fft_size,
                            fft_strides,
                            output_fft_strides);
    }
}

void fft_postprocessing(ov::TensorVector& outputs,
                        const ov::element::Type output_type,
                        const std::vector<float>& fft_result) {
    size_t fft_result_size = fft_result.size();

    switch (output_type) {
    case element::Type_t::bf16: {
        bfloat16* result_ptr = outputs[0].data<bfloat16>();
        for (size_t i = 0; i < fft_result_size; ++i) {
            result_ptr[i] = bfloat16(fft_result[i]);
        }
    } break;
    case element::Type_t::f16: {
        float16* result_ptr = outputs[0].data<float16>();
        for (size_t i = 0; i < fft_result_size; ++i) {
            result_ptr[i] = float16(fft_result[i]);
        }
    } break;
    case element::Type_t::f32: {
        float* result_ptr = outputs[0].data<float>();
        memcpy(result_ptr, fft_result.data(), fft_result_size * sizeof(float));
    } break;
    default:;
    }
}
}  // namespace reference
}  // namespace ov
