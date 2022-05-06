//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <utility>
#include <vector>

#include "ngraph/runtime/reference/fft.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace ngraph::runtime::reference;

namespace ngraph {
namespace runtime {
namespace reference {
namespace {
using complex_type = std::complex<float>;

// TODO: Can be moved to common?
// TODO: Can be unified with strides_for_outer_axes?
Shape shape_for_outer_axes(const Shape& shape, int64_t inner_axis) {
    Shape result;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != static_cast<size_t>(inner_axis)) {
            result.push_back(shape[i]);
        }
    }
    return result;
}

std::vector<int64_t> strides_for_outer_axes(const std::vector<int64_t>& strides, int64_t inner_axis) {
    std::vector<int64_t> result;
    for (size_t i = 0; i < strides.size(); ++i) {
        if (i != static_cast<size_t>(inner_axis)) {
            result.push_back(strides[i]);
        }
    }
    return result;
}

// When we reverted shape, we need to revert FFT axes.
std::vector<int64_t> reverse_fft_axes(const std::vector<int64_t>& axes, int64_t complex_data_rank) {
    auto result = axes;
    for (int64_t& axis : result) {
        axis = complex_data_rank - 1 - axis;
    }
    return result;
}

// TODO: Check if can be removed
std::vector<int64_t> get_outer_fft_axes(const std::vector<int64_t>& v) {
    if (v.empty() || v.size() == 1) {
        return {};
    }

    return std::vector<int64_t>(v.begin(), v.end() - 1);
}

std::vector<complex_type> extend_input_data(const std::vector<float>& input_data,
                                            const Shape& input_data_shape,
                                            const Shape& shape_of_extended_input_data,
                                            const std::vector<int64_t>& axes_data,  // really needed?
                                            const int64_t last_signal_size) {
    std::vector<complex_type> extended_input_complex(shape_size(shape_of_extended_input_data) / 2, complex_type{0, 0});

    const complex_type* complex_input_ptr = reinterpret_cast<const complex_type*>(input_data.data());
    const auto reversed_input_data_shape = fft_common::reverse_shape_of_emulated_complex_tensor(input_data_shape);
    const auto input_data_strides = fft_common::compute_strides(reversed_input_data_shape);

    const auto reversed_ext_input_data_shape =
        fft_common::reverse_shape_of_emulated_complex_tensor(shape_of_extended_input_data);
    const auto reversed_ext_input_strides = fft_common::compute_strides(reversed_ext_input_data_shape);
    const auto last_axis = axes_data.back();
    const auto outer_extended_shape = shape_for_outer_axes(shape_of_extended_input_data, last_axis);
    const auto reversed_outer_extended_shape =
        fft_common::reverse_shape_of_emulated_complex_tensor(outer_extended_shape);
    const auto outer_extended_shape_strides = fft_common::compute_strides(reversed_outer_extended_shape);
    const auto outer_extended_size = outer_extended_shape_strides.back();

    const auto complex_data_rank = static_cast<int64_t>(reversed_ext_input_data_shape.size());
    const auto reversed_axes = reverse_fft_axes(axes_data, complex_data_rank);
    const auto reversed_last_axis = reversed_axes.back();
    const auto outer_strides = strides_for_outer_axes(input_data_strides, reversed_last_axis);

    const auto outer_extended_strides = strides_for_outer_axes(reversed_ext_input_strides, reversed_last_axis);
    const auto inner_stride = input_data_strides[reversed_last_axis];
    const auto inner_extended_stride = reversed_ext_input_strides[reversed_last_axis];

    const auto inner_bound = last_signal_size / 2 + 1;
    const auto inner_size = reversed_input_data_shape[reversed_last_axis];
    const auto inner_extended_size = reversed_ext_input_data_shape[reversed_last_axis];

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

std::pair<std::vector<float>, Shape> calculate_idft_with_respect_to_given_axis(const std::vector<float>& input_data,
                                                                               const Shape& input_shape,
                                                                               const int64_t axis,
                                                                               const int64_t signal_size) {
    auto output_shape = input_shape;
    output_shape[axis] = signal_size;
    auto result = std::vector<float>(shape_size(output_shape), 0.0f);

    fft(reinterpret_cast<const float*>(input_data.data()),
        input_shape,
        std::vector<int64_t>{axis}.data(),
        Shape{1},
        result.data(),
        output_shape,
        FFTKind::Inverse);

    return std::make_pair(result, output_shape);
}

}  // namespace

void irdft(const std::vector<float>& input_data,
           const Shape& input_data_shape,
           const std::vector<int64_t>& axes_data,
           float* irdft_result,
           const Shape& fft_output_shape,
           int64_t last_signal_size) {  // TODO: seems to be not needed

    std::vector<float> float_data = input_data;
    const auto outer_fft_axes = get_outer_fft_axes(axes_data);
    Shape float_data_shape = input_data_shape;
    for (const auto a : outer_fft_axes) {  // TODO CHECK NEGATIVE AXES
        std::tie(float_data, float_data_shape) =
            calculate_idft_with_respect_to_given_axis(float_data, float_data_shape, a, fft_output_shape[a]);
    }

    const auto last_axis = axes_data.back();
    auto shape_of_extended_input_data = float_data_shape;
    shape_of_extended_input_data[last_axis] = last_signal_size;
    std::vector<complex_type> ifft_result(shape_size(fft_output_shape) / 2);
    std::vector<complex_type> extended_input_complex =
        extend_input_data(float_data, float_data_shape, shape_of_extended_input_data, axes_data, last_signal_size);
    fft(reinterpret_cast<const float*>(extended_input_complex.data()),
        shape_of_extended_input_data,
        std::vector<int64_t>{last_axis}.data(),
        Shape{1},
        reinterpret_cast<float*>(ifft_result.data()),
        fft_output_shape,
        FFTKind::Inverse);

    for (size_t i = 0; i < ifft_result.size(); ++i) {
        irdft_result[i] = std::real(ifft_result[i]);
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph