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

#include "ngraph/runtime/reference/irdft.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <iostream>
#include <ngraph/runtime/reference/utils/fft_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace ngraph::runtime::reference;

namespace ngraph {
namespace runtime {
namespace reference {
namespace {
using complex_type = std::complex<float>;

// When we reverted shape, we need to revert IRDFT axes.
std::vector<int64_t> reverse_fft_axes(const std::vector<int64_t>& axes, int64_t complex_data_rank) {
    auto result = axes;
    for (int64_t& axis : result) {
        axis = complex_data_rank - 1 - axis;
    }
    return result;
}

// Helper function to get only length with respect to given axes.
std::vector<int64_t> get_lengths(const std::vector<int64_t>& shape, const std::vector<int64_t>& axes) {
    std::vector<int64_t> lengths;
    for (int64_t axis : axes) {
        lengths.push_back(shape[axis]);
    }
    return lengths;
}

// This function calculates 'outer axes', that is axes that are not transformed by IRDFT.
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

// Copying input data to the given memory domain. Returns true if the copied blob is zero, and false otherwise.
bool copy_data_from_input_and_check_is_blob_zero(complex_type* result,
                                                 const complex_type* input_data,
                                                 int64_t src_index,
                                                 int64_t fft_size,
                                                 const std::vector<int64_t>& fft_strides,
                                                 const std::vector<int64_t>& input_fft_lengths,
                                                 const std::vector<int64_t>& input_fft_strides,
                                                 int64_t last_axis_upper_bound) {
    bool blob_is_zero = true;
    for (int64_t idx = 0; idx < fft_size; ++idx) {
        auto coords = fft_common::coords_from_index(idx, fft_strides);
        if (coords.back() >= last_axis_upper_bound) {
            continue;
        }
        complex_type value = get_value_from_input(input_data, src_index, coords, input_fft_lengths, input_fft_strides);
        result[idx] = value;
        blob_is_zero = blob_is_zero && (value == complex_type{0.0f, 0.0f});
    }
    return blob_is_zero;
}

template <typename T>
void print_vector(const std::vector<T>& v, const std::string& prefix) {
    std::cout << prefix;
    for (const auto& x : v) {
        std::cout << x << " ";
    }
    std::cout << "\n";
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

std::vector<int64_t> lengths_except_given_axis(const std::vector<int64_t>& v, int64_t axis) {
    auto result = v;
    if (axis >= 0 && axis < static_cast<int64_t>(v.size()))
        result.erase(result.begin() + axis);
    return result;
}

enum class FFTKind { Forward, Inverse };

// This function gathers data from the input of 1D FFT to the contiguous buffer. Returns true
// if copied blob is zero, and false otherwise.
bool gather_to_buffer(const complex_type* data,
                      int64_t length,
                      int64_t start,
                      int64_t stride,
                      complex_type* buffer,
                      bool expand_input) {
    bool blob_is_zero = true;
    const int64_t ub = expand_input ? length / 2 + 1 : length;
    for (int64_t k = 0; k < ub; ++k) {
        complex_type value = data[start + k * stride];
        buffer[k] = value;
        blob_is_zero = blob_is_zero && (value == complex_type{0.0f, 0.0f});
        if (expand_input) {
            // Use conjugates of the values at indices [1 ... (ub - 2)] when the
            // length is even and at indices [1 ... (ub - 1)] when the length is odd
            // to calculate missing values at indices [(length - 1) ... ub].
            if (k > 0 && k < (length - ub + 1)) {
                buffer[length - k] = std::conj(value);
            }
        }
    }
    return blob_is_zero;
}

static constexpr float pi = 3.141592653589793238462643f;

// This function calculates e^{-2i\pi k / length} for the forward FFT, and
// e^{2i\pi k / length} otherwise. Here 'i' is an imaginary unit.
complex_type twiddle(int64_t k, int64_t length, FFTKind fft_kind) {
    float angle = -2.0f * pi * static_cast<float>(k) / static_cast<float>(length);
    complex_type result = std::exp(complex_type(0.0f, angle));
    return (fft_kind == FFTKind::Inverse) ? std::conj(result) : result;
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
                     FFTKind fft_kind,
                     bool expand_input) {
    bool gathered_blob_is_zero = gather_to_buffer(data, length, fft_offset, stride, buffer, expand_input);
    if (gathered_blob_is_zero) {
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
            value /= complex_type(length, 0.0f);
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
                 FFTKind fft_kind,
                 bool expand_input) {
    bool gathered_blob_is_zero = gather_to_buffer(data, length, fft_offset, stride, buffer, expand_input);
    if (gathered_blob_is_zero) {
        return;
    }

    for (int64_t k = 0; k < length; ++k) {
        complex_type value = complex_type(0.0f, 0.0f);
        for (int64_t n = 0; n < length; ++n) {
            value += buffer[n] * twiddle(n * k, length, fft_kind);
        }
        if (fft_kind == FFTKind::Inverse) {
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
           FFTKind fft_kind,
           bool expand_input) {
    if (is_power_of_two(length)) {
        optimized_fft1d(length, fft_offset, stride, data, buffer, fft_kind, expand_input);
    } else {
        naive_fft1d(length, fft_offset, stride, data, buffer, fft_kind, expand_input);
    }
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

// Calculation of IRDFT
void irdft_calculation(const float* input_data,
                       const Shape& input_data_shape,
                       const std::vector<int64_t>& axes_data,
                       float* fft_result,
                       const Shape& fft_output_shape) {
    std::cout << "We are in the function irdft_calculation()...\n";
    std::cout << "input_data_shape: " << input_data_shape << "\n";
    std::cout << "fft_output_shape: " << fft_output_shape << "\n";

    const complex_type* complex_input_data_ptr = reinterpret_cast<const complex_type*>(input_data);
    complex_type* complex_output_ptr = reinterpret_cast<complex_type*>(fft_result);
    std::cout << "input_data pointer:  " << complex_input_data_ptr << "\n";
    std::cout << "output data pointer: " << complex_output_ptr << "\n";

    print_vector(axes_data, "axes_data: ");

    const int64_t complex_data_rank = static_cast<int64_t>(input_data_shape.size()) - 1;
    std::cout << "complex_data_rank: " << complex_data_rank << "\n";
    const auto fft_axes = reverse_fft_axes(axes_data, complex_data_rank);
    print_vector(fft_axes, "fft_axes: ");

    const auto reversed_output_shape = fft_common::reverse_shape_of_emulated_complex_tensor(fft_output_shape);
    print_vector(reversed_output_shape, "reversed_output_shape: ");

    const int64_t fft_rank = fft_axes.size();
    std::cout << "fft_rank: " << fft_rank << "\n";

    const auto fft_lengths = get_lengths(reversed_output_shape, fft_axes);
    print_vector(fft_lengths, "fft_lengths: ");

    const auto fft_strides = fft_common::compute_strides(fft_lengths);
    print_vector(fft_strides, "fft_strides: ");

    const int64_t fft_size = fft_strides[fft_rank];
    std::cout << "fft_size: " << fft_size << "\n";

    if (fft_size <= 0) {
        return;
    }

    const int64_t buffer_size = compute_buffer_size(fft_lengths);
    std::cout << "buffer_size: " << buffer_size << "\n";

    std::vector<complex_type> data(fft_size);
    std::vector<complex_type> buffer(buffer_size);

    const auto outer_axes = get_outer_axes(fft_axes, complex_data_rank);
    print_vector(outer_axes, "outer_axes: ");

    const int64_t outer_rank = outer_axes.size();
    std::cout << "outer_rank: " << outer_rank << "\n";

    const auto outer_lengths = get_lengths(reversed_output_shape, outer_axes);
    print_vector(outer_lengths, "outer_lengths: ");

    const auto outer_strides = fft_common::compute_strides(outer_lengths);
    print_vector(outer_strides, "outer_strides: ");

    const int64_t outer_size = outer_strides[outer_rank];
    std::cout << "outer_size: " << outer_size << "\n";

    const auto output_strides = fft_common::compute_strides(reversed_output_shape);
    const auto output_fft_strides = get_lengths(output_strides, fft_axes);
    const auto output_outer_strides = get_lengths(output_strides, outer_axes);
    print_vector(output_strides, "output_strides: ");
    print_vector(output_fft_strides, "output_fft_strides: ");
    print_vector(output_outer_strides, "output_outer_strides: ");

    const auto reversed_input_shape = fft_common::reverse_shape_of_emulated_complex_tensor(input_data_shape);
    const auto input_fft_lengths = get_lengths(reversed_input_shape, fft_axes);
    const auto input_strides = fft_common::compute_strides(reversed_input_shape);
    const auto input_fft_strides = get_lengths(input_strides, fft_axes);
    const auto input_outer_strides = get_lengths(input_strides, outer_axes);
    print_vector(reversed_input_shape, "reversed_input_shape: ");
    print_vector(input_fft_lengths, "input_fft_lengths: ");
    print_vector(input_strides, "input_strides: ");
    print_vector(input_fft_strides, "input_fft_strides: ");
    print_vector(input_outer_strides, "input_outer_strides: ");

    const int64_t last_axis_upper_bound = fft_lengths.back() / 2 + 1;
    std::cout << "last_axis_upper_bound: " << last_axis_upper_bound << "\n";
    // Loop along with 'outer' dimensions, that is along with
    // not transformed dimensions.
    for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
        std::cout << "************************************************************\n";
        const auto outer_coords = fft_common::coords_from_index(outer_idx, outer_strides);
        int64_t outer_input_offset = fft_common::offset_from_coords_and_strides(outer_coords, input_outer_strides);
        print_vector(outer_coords, "    outer_coords: ");
        std::cout << "    outer_input_offset: " << outer_input_offset << "\n";

        // Copying current data to transform
        bool blob_is_zero = copy_data_from_input_and_check_is_blob_zero(data.data(),
                                                                        complex_input_data_ptr,
                                                                        outer_input_offset,
                                                                        fft_size,
                                                                        fft_strides,
                                                                        input_fft_lengths,
                                                                        input_fft_strides,
                                                                        last_axis_upper_bound);
        std::cout << (blob_is_zero ? "    current blob is zero\n" : "    current blob is not zero\n");
        if (!blob_is_zero) {
            // The loop along with all transformed axes.
            for (int64_t axis_idx = 0; axis_idx < fft_rank; ++axis_idx) {
                std::cout << "        axis_idx:           " << axis_idx << "\n";
                int64_t current_fft_stride = fft_strides[axis_idx];
                int64_t current_fft_length = fft_lengths[axis_idx];
                std::cout << "        current_fft_stride: " << current_fft_stride << "\n";
                std::cout << "        current_fft_length: " << current_fft_length << "\n";

                auto outer_fft_lengths = lengths_except_given_axis(fft_lengths, axis_idx);
                auto outer_fft_axes = lengths_except_given_axis(fft_axes, axis_idx);
                int64_t outer_fft_size = fft_size / current_fft_length;
                if (axis_idx != fft_rank - 1) {
                    outer_fft_size = (outer_fft_size / fft_lengths.back()) * (fft_lengths.back() / 2 + 1);
                }
                print_vector(outer_fft_lengths, "        outer_fft_lengths: ");
                print_vector(outer_fft_axes, "        outer_fft_axes: ");
                std::cout << "        outer_fft_size:     " << outer_fft_size << "\n";

                auto outer_fft_strides = fft_common::compute_strides(outer_fft_lengths);
                auto fft_strides_for_outer_fft_axes = lengths_except_given_axis(fft_strides, axis_idx);
                print_vector(outer_fft_strides, "        outer_fft_strides: ");
                print_vector(fft_strides_for_outer_fft_axes, "        fft_strides_for_outer_fft_axes: ");

                // Loop along with all FFT axes, except the current one.
                for (int64_t outer_fft_idx = 0; outer_fft_idx < outer_fft_size; ++outer_fft_idx) {
                    std::cout << "            outer_fft_idx: " << outer_fft_idx << "\n";
                    const auto outer_fft_coords = fft_common::coords_from_index(outer_fft_idx, outer_fft_strides);
                    print_vector(outer_fft_coords, "            outer_fft_coords: ");
                    int64_t outer_fft_offset =
                        fft_common::offset_from_coords_and_strides(outer_fft_coords, fft_strides_for_outer_fft_axes);
                    std::cout << "            outer_fft_offset: " << outer_fft_offset << "\n";
                    // Calculation of 1D FFT
                    fft1d(current_fft_length,
                          outer_fft_offset,
                          current_fft_stride,
                          data.data(),
                          buffer.data(),
                          FFTKind::Inverse,
                          axis_idx == fft_rank - 1);
                }
            }
        }

        // Copying current calculated data to the output blob.
        int64_t outer_output_offset = fft_common::offset_from_coords_and_strides(outer_coords, output_outer_strides);
        std::cout << "    outer_output_offset: " << outer_output_offset << "\n";
        copy_data_to_output(complex_output_ptr,
                            data.data(),
                            outer_output_offset,
                            fft_size,
                            fft_strides,
                            output_fft_strides);
    }
}

void irdft_postprocessing(const complex_type* intermediate_results,
                          float* results,
                          const Shape& output_shape) {
    const size_t output_size = shape_size(output_shape);
    for (size_t i = 0; i < output_size; ++i) {
        results[i] = std::real(intermediate_results[i]);
    }
}
}  // namespace

void irdft(const float* input_data,
           const Shape& input_data_shape,
           const std::vector<int64_t>& axes_data,
           float* fft_result,
           const Shape& fft_output_shape,
           const Shape& output_shape) {
    std::vector<complex_type> intermediate_results(shape_size(fft_output_shape) / 2);
    irdft_calculation(input_data,
                      input_data_shape,
                      axes_data,
                      reinterpret_cast<float*>(intermediate_results.data()),
                      fft_output_shape);
    irdft_postprocessing(intermediate_results.data(), fft_result, output_shape);
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
