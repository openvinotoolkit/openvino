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

// When we reverted shape, we need to revert FFT axes.
std::vector<int64_t> reverse_fft_axes(const std::vector<int64_t>& axes, int64_t complex_data_rank) {
    auto result = axes;
    for (int64_t& axis : result) {
        axis = complex_data_rank - 1 - axis;
    }
    return result;
}

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
}  // namespace

void irdft(const std::vector<float>& input_data,
           const Shape& input_data_shape,
           const std::vector<int64_t>& axes_data,
           const Shape& output_ifft_shape,
           const int64_t last_signal_size,
           float* irdft_result) {
    std::cout << "We are in the reference for IRDFT.\n";
    std::cout << "input_data_shape:  " << input_data_shape << "\n";
    std::cout << "output_ifft_shape: " << output_ifft_shape << "\n";
    std::cout << "last_signal_size:  " << last_signal_size << "\n";
    std::cout << "axes_data: ";
    for (const auto a : axes_data) {
        std::cout << a << ", ";
    }
    std::cout << "\n";
    const auto last_axis = axes_data.back();
    std::cout << "last axis: " << last_axis << "\n";
    auto shape_of_extended_input_data = input_data_shape;
    shape_of_extended_input_data[last_axis] = last_signal_size;
//     auto shape_of_extended_complex_input_data = shape_of_extended_input_data;
//     shape_of_extended_complex_input_data.pop_back();
    std::cout << "shape_of_extended_input_data: " << shape_of_extended_input_data << "\n";
//     std::cout << "shape_of_extended_complex_input_data: " << shape_of_extended_complex_input_data << "\n";

    const auto reversed_input_data_shape = fft_common::reverse_shape(input_data_shape);
    const auto reversed_extended_input_data_shape = fft_common::reverse_shape(shape_of_extended_input_data);
    std::cout << "reversed_input_data_shape: ";
    for (const auto d : reversed_input_data_shape) {
        std::cout << d << " ";
    }
    std::cout << "\n";
    std::cout << "reversed_extended_input_data_shape: ";
    for (const auto d : reversed_extended_input_data_shape) {
        std::cout << d << " ";
    }
    std::cout << "\n";

    const int64_t complex_data_rank = static_cast<int64_t>(reversed_extended_input_data_shape.size());
    const auto reversed_axes = reverse_fft_axes(axes_data, complex_data_rank);
    std::cout << "complex_data_rank: " << complex_data_rank << "\n";
    std::cout << "reversed_axes: ";
    for (const auto d : reversed_axes) {
        std::cout << d << " ";
    }
    std::cout << "\n";
    const auto reversed_last_axis = reversed_axes.back();
    std::cout << "reversed_last_axis: " << reversed_last_axis << "\n";

    const auto input_data_strides = fft_common::compute_strides(reversed_input_data_shape);
    const auto extended_input_data_strides = fft_common::compute_strides(reversed_extended_input_data_shape);
    std::cout << "input_data_strides: ";
    for (const auto d : input_data_strides) {
        std::cout << d << " ";
    }
    std::cout << "\n";
    std::cout << "extended_input_data_strides: ";
    for (const auto d : extended_input_data_strides) {
        std::cout << d << " ";
    }
    std::cout << "\n";

    std::vector<complex_type> extended_input_data(extended_input_data_strides.back());
    std::cout << "number of elements in complex extended input data: " << extended_input_data_strides.back() << "\n";

    const auto outer_extended_shape = shape_for_outer_axes(shape_of_extended_input_data, last_axis);
    std::cout << "outer_extended_shape: " << outer_extended_shape << "\n";

    const auto outer_shape = shape_for_outer_axes(input_data_shape, last_axis);
    std::cout << "outer_shape: " << outer_shape << "\n";

    const auto reversed_outer_extended_shape = fft_common::reverse_shape(outer_extended_shape);
    std::cout << "reversed_outer_extended_shape: ";
    for (const auto d : reversed_outer_extended_shape) {
        std::cout << d << " ";
    }
    std::cout << "\n";
    const auto outer_extended_shape_strides = fft_common::compute_strides(reversed_outer_extended_shape);
    std::cout << "outer_extended_shape_strides: ";
    for (const auto d : outer_extended_shape_strides) {
        std::cout << d << " ";
    }
    std::cout << "\n";
    const auto outer_extended_size = outer_extended_shape_strides.back();
    std::cout << "outer_extended_size: " << outer_extended_size << "\n";

    const auto reversed_outer_shape = fft_common::reverse_shape(outer_shape);
    std::cout << "reversed_outer_shape: ";
    for (const auto d : reversed_outer_shape) {
        std::cout << d << " ";
    }
    std::cout << "\n";
    const complex_type* complex_input_data_ptr = reinterpret_cast<const complex_type*>(input_data.data());
    complex_type* extended_data_ptr = extended_input_data.data();

    const auto inner_extended_size = reversed_extended_input_data_shape[reversed_last_axis];
    std::cout << "inner_extended_size: " << inner_extended_size << "\n";

    const auto inner_size = reversed_input_data_shape[reversed_last_axis];
    std::cout << "inner_size: " << inner_size << "\n";

    const auto inner_bound = last_signal_size / 2 + 1;
    std::cout << "inner_bound: " << inner_bound << "\n";

    const auto outer_strides = strides_for_outer_axes(input_data_strides, reversed_last_axis);
    const auto outer_extended_strides = strides_for_outer_axes(extended_input_data_strides, reversed_last_axis);
    std::cout << "outer_strides: ";
    for (const auto d : outer_strides) {
        std::cout << d << " ";
    }
    std::cout << "\n";
    std::cout << "outer_extended_strides: ";
    for (const auto d : outer_extended_strides) {
        std::cout << d << " ";
    }
    std::cout << "\n";
    const auto inner_stride = input_data_strides[reversed_last_axis];
    const auto inner_extended_stride = extended_input_data_strides[reversed_last_axis];
    std::cout << "inner_stride: " << inner_stride << "\n";
    std::cout << "inner_extended_stride: " << inner_extended_stride << "\n";

    for (int64_t i = 0; i < outer_extended_size; ++i) {
        const auto outer_coords = fft_common::coords_from_index(i, outer_extended_shape_strides);
        std::cout << "outer_coords: ";
        for (const auto d : outer_coords) {
            std::cout << d << " ";
        }
        std::cout << "\n";
        const auto outer_input_offset = fft_common::offset_from_coords_and_strides(outer_coords, outer_strides);
        std::cout << "outer_input_offset: " << outer_input_offset << "\n";
        const auto outer_output_offset = fft_common::offset_from_coords_and_strides(outer_coords, outer_extended_strides);
        std::cout << "outer_output_offset: " << outer_output_offset << "\n";

        for (int64_t j = 0; j < inner_bound; ++j) {
            complex_type value = complex_type(0.0, 0.0);
            if (j < inner_size) {
                value = complex_input_data_ptr[outer_input_offset + j * inner_stride];
            }
            extended_data_ptr[outer_output_offset + j * inner_extended_stride] = value;
            if (j > 0 && j < (inner_extended_size - inner_bound + 1)) {
                extended_data_ptr[outer_output_offset + (inner_extended_size - j) * inner_extended_stride] = std::conj(value);
            }
        }

//         for (int64_t j = 0; j < inner_extended_size; ++j) {
//             complex_type extended_elem;
//             if (j < inner_size) {
//                 extended_elem = complex_input_data_ptr[outer_input_offset + j * inner_stride];
//             } else {
//                 extended_elem = std::conj(complex_input_data_ptr[outer_input_offset + (inner_extended_size - j) * inner_stride]);
//             }
//             extended_data_ptr[outer_output_offset + j * inner_extended_stride] = extended_elem;
//         }
    }
    std::cout << "extended_input_data: [";
    for (std::size_t i = 0; i < extended_input_data.size(); ++i) {
        std::cout << std::real(extended_input_data[i]) << ", " << std::imag(extended_input_data[i]) << ", ";
    }
    std::cout << "]\n";

    std::vector<complex_type> ifft_result(shape_size(output_ifft_shape) / 2);

    // Here calculation of IDFT
    fft(reinterpret_cast<const float*>(extended_input_data.data()),
        shape_of_extended_input_data,
        axes_data.data(),
        Shape{axes_data.size()},
        reinterpret_cast<float*>(ifft_result.data()),
        output_ifft_shape,
        FFTKind::Inverse);

    std::cout << "fft calculation result: ";
    for (const auto x : ifft_result) {
        std::cout << x << ", ";
    }
    std::cout << "\n";

    for (size_t i = 0; i < ifft_result.size(); ++i) {
        irdft_result[i] = std::real(ifft_result[i]);
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph