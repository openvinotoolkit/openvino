// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/irdft.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/fft.hpp"
#include "openvino/reference/utils/fft_common.hpp"

namespace ov {
namespace reference {
namespace {
using complex_type = std::complex<float>;

// Remove element (dimension from shape or value from stride) on the given position
template <typename T>
std::vector<T> remove_from_position(const std::vector<T>& vec, const int64_t pos) {
    assert(vec.size() > static_cast<size_t>(pos));
    auto result = vec;
    result.erase(std::begin(result) + pos);
    return result;
}

// Returns all axes except the last one (or empty vector for the size less than 2)
std::vector<int64_t> get_outer_fft_axes(const std::vector<int64_t>& vec) {
    if (vec.size() < 2) {
        return {};
    }
    return std::vector<int64_t>(vec.begin(), vec.end() - 1);
}

// RDFT operation (which consumes only real inputs) returns Hermitian-symmetric results.
// It means that part of values are just the complex conjugates of the values from the corresponding symmetric
// positions. IRDFT as the inverse operation to RDFT has to put these symmetric values in order to calculate correct
// results.
std::vector<complex_type> extend_to_hermitian_symmetric(const std::vector<float>& input_data,
                                                        const Shape& input_data_shape,
                                                        const Shape& shape_of_extended_input_data,
                                                        const std::vector<int64_t>& axes_data,
                                                        const int64_t last_signal_size) {
    const complex_type* complex_input_ptr = reinterpret_cast<const complex_type*>(input_data.data());
    const auto reversed_input_data_shape = fft_common::reverse_shape_of_emulated_complex_tensor(input_data_shape);
    const auto input_data_strides = fft_common::compute_strides(reversed_input_data_shape);

    const auto reversed_ext_input_data_shape =
        fft_common::reverse_shape_of_emulated_complex_tensor(shape_of_extended_input_data);
    const auto reversed_ext_input_strides = fft_common::compute_strides(reversed_ext_input_data_shape);
    Shape outer_extended_shape = shape_of_extended_input_data;
    if (shape_of_extended_input_data.size() == 2) {
        outer_extended_shape[axes_data.back()] = 1;
    } else {
        outer_extended_shape = remove_from_position(shape_of_extended_input_data, axes_data.back());
    }
    const auto reversed_outer_extended_shape =
        fft_common::reverse_shape_of_emulated_complex_tensor(outer_extended_shape);
    const auto outer_extended_shape_strides = fft_common::compute_strides(reversed_outer_extended_shape);
    const auto outer_extended_size = outer_extended_shape_strides.back();

    const auto complex_data_rank = static_cast<int64_t>(reversed_ext_input_data_shape.size());
    const auto reversed_axes = fft_common::reverse_fft_axes(axes_data, complex_data_rank);
    const auto reversed_last_axis = reversed_axes.back();
    const auto outer_strides = remove_from_position(input_data_strides, reversed_last_axis);

    const auto outer_extended_strides = remove_from_position(reversed_ext_input_strides, reversed_last_axis);
    const auto inner_stride = input_data_strides[reversed_last_axis];
    const auto inner_extended_stride = reversed_ext_input_strides[reversed_last_axis];

    const auto inner_bound = last_signal_size / 2 + 1;
    const auto inner_size = reversed_input_data_shape[reversed_last_axis];
    const auto inner_extended_size = reversed_ext_input_data_shape[reversed_last_axis];

    std::vector<complex_type> extended_input_complex(shape_size(shape_of_extended_input_data) / 2, complex_type{0, 0});
    for (int64_t i = 0; i < outer_extended_size; ++i) {
        const auto outer_coords = fft_common::coords_from_index(i, outer_extended_shape_strides);
        const auto outer_input_offset = fft_common::offset_from_coords_and_strides(outer_coords, outer_strides);
        const auto outer_output_offset =
            fft_common::offset_from_coords_and_strides(outer_coords, outer_extended_strides);

        for (int64_t j = 0; j < inner_bound; ++j) {
            complex_type value = complex_type(0.0, 0.0);
            if (j < inner_size) {
                value = complex_input_ptr[outer_input_offset + j * inner_stride];
            }
            extended_input_complex[outer_output_offset + j * inner_extended_stride] = value;
            if (j > 0 && j < (inner_extended_size - inner_bound + 1)) {
                extended_input_complex[outer_output_offset + (inner_extended_size - j) * inner_extended_stride] =
                    std::conj(value);
            }
        }
    }
    return extended_input_complex;
}

}  // namespace

void irdft(const std::vector<float>& input_data,
           const Shape& input_data_shape,
           const std::vector<int64_t>& axes_data,
           float* irdft_result,
           const Shape& fft_output_shape,
           const Shape& irdft_output_shape,
           const int64_t last_signal_size) {
    // calculate inverse FFT over the outer axes
    const auto outer_ifft_axes = get_outer_fft_axes(axes_data);
    auto outer_ifft_shape = input_data_shape;
    for (const auto& a : outer_ifft_axes) {
        outer_ifft_shape[a] = fft_output_shape[a];
    }
    std::vector<float> outer_fft_result(shape_size(outer_ifft_shape), 0.0f);
    fft(reinterpret_cast<const float*>(input_data.data()),
        input_data_shape,
        outer_ifft_axes.data(),
        Shape{outer_ifft_axes.size()},
        outer_fft_result.data(),
        outer_ifft_shape,
        FFTKind::Inverse);

    // adjust the input to Hermitian-symmetric size
    const auto last_axis = axes_data.back();
    auto extended_data_shape = outer_ifft_shape;
    extended_data_shape[last_axis] = last_signal_size;
    const auto complex_ifft_result_size = shape_size(irdft_output_shape);
    std::vector<complex_type> complex_ifft_result(complex_ifft_result_size);
    std::vector<complex_type> extended_complex_data = extend_to_hermitian_symmetric(outer_fft_result,
                                                                                    outer_ifft_shape,
                                                                                    extended_data_shape,
                                                                                    axes_data,
                                                                                    last_signal_size);

    // calculate inverse FFT on adjusted data over the last last axis
    fft(reinterpret_cast<const float*>(extended_complex_data.data()),
        extended_data_shape,
        &last_axis,
        Shape{1},
        reinterpret_cast<float*>(complex_ifft_result.data()),
        fft_output_shape,
        FFTKind::Inverse);

    // cut out the imaginary part of the complex result
    for (size_t i = 0; i < complex_ifft_result_size; ++i) {
        irdft_result[i] = std::real(complex_ifft_result[i]);
    }
}
}  // namespace reference
}  // namespace ov
