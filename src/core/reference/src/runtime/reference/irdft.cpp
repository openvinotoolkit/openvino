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

template<typename T>
std::vector<T> get_reversed_vector(const std::vector<T>& v) {
    std::vector<T> result = v;
    if(v.empty())
    {
        return result;
    }
    size_t i = 0;
    size_t j = v.size() - 1;
    while(i < j)
    {
        using namespace std;
        swap(result[i], result[j]);
        i++;
        j--;
    }
    return result;
}

std::vector<int64_t> get_outer_fft_axes(const std::vector<int64_t>& v) {
    if(v.empty() || v.size() == 1) {
        return {};
    }

    return std::vector<int64_t>(v.rbegin(), v.rend() - 1);
}

struct Float_data {
    std::vector<float> data;
    Shape shape;
};

Float_data calculate_idft_with_respect_to_given_axis(const Float_data& input_data,
                                                     const int64_t axis,
                                                     const int64_t signal_size)
{
    Float_data result;

    auto output_shape = input_data.shape;
    output_shape[axis] = signal_size;

    result.shape = output_shape;
    result.data = std::vector<float>(shape_size(output_shape), 0.0f);

    const std::vector<int64_t> axes_data = {axis};

    fft(input_data.data.data(),
        input_data.shape,
        axes_data.data(),
        Shape{axes_data.size()},
        result.data.data(),
        output_shape,
        FFTKind::Inverse);

    return result;
}

struct Complex_data {
    std::vector<complex_type> data;
    Shape shape;
};

Complex_data extend_input_data(const std::vector<float>& input_data,
                               const Shape& input_data_shape,
                               const std::vector<int64_t>& axes_data,
                               const int64_t last_signal_size) {
    Complex_data result;

    const auto last_axis = axes_data.back();
    auto shape_of_extended_input_data = input_data_shape;
    shape_of_extended_input_data[last_axis] = last_signal_size;

    const auto reversed_input_data_shape = fft_common::reverse_shape(input_data_shape);
    const auto reversed_extended_input_data_shape = fft_common::reverse_shape(shape_of_extended_input_data);

    const int64_t complex_data_rank = static_cast<int64_t>(reversed_extended_input_data_shape.size());
    const auto reversed_axes = reverse_fft_axes(axes_data, complex_data_rank);

    const auto reversed_last_axis = reversed_axes.back();

    const auto input_data_strides = fft_common::compute_strides(reversed_input_data_shape);
    const auto extended_input_data_strides = fft_common::compute_strides(reversed_extended_input_data_shape);

    result.data = std::vector<complex_type>(extended_input_data_strides.back());
    result.shape = shape_of_extended_input_data;

    const auto outer_extended_shape = shape_for_outer_axes(shape_of_extended_input_data, last_axis);
    const auto outer_shape = shape_for_outer_axes(input_data_shape, last_axis);
    const auto reversed_outer_extended_shape = fft_common::reverse_shape(outer_extended_shape);
    const auto outer_extended_shape_strides = fft_common::compute_strides(reversed_outer_extended_shape);
    const auto outer_extended_size = outer_extended_shape_strides.back();
    const auto reversed_outer_shape = fft_common::reverse_shape(outer_shape);

    const complex_type* complex_input_data_ptr = reinterpret_cast<const complex_type*>(input_data.data());
    complex_type* extended_data_ptr = result.data.data();

    const auto inner_extended_size = reversed_extended_input_data_shape[reversed_last_axis];
    const auto inner_size = reversed_input_data_shape[reversed_last_axis];
    const auto inner_bound = last_signal_size / 2 + 1;
    const auto outer_strides = strides_for_outer_axes(input_data_strides, reversed_last_axis);
    const auto outer_extended_strides = strides_for_outer_axes(extended_input_data_strides, reversed_last_axis);
    const auto inner_stride = input_data_strides[reversed_last_axis];
    const auto inner_extended_stride = extended_input_data_strides[reversed_last_axis];

    for (int64_t i = 0; i < outer_extended_size; ++i) {
        const auto outer_coords = fft_common::coords_from_index(i, outer_extended_shape_strides);
        const auto outer_input_offset = fft_common::offset_from_coords_and_strides(outer_coords, outer_strides);
        const auto outer_output_offset = fft_common::offset_from_coords_and_strides(outer_coords, outer_extended_strides);

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

    const auto outer_fft_axes = get_outer_fft_axes(axes_data);
    std::cout << "outer_fft_axes: ";
    for (const auto a : outer_fft_axes) {
        std::cout << a << ", ";
    }
    std::cout << "\n";

    const auto extended_input_data =  extend_input_data(input_data, input_data_shape, axes_data, last_signal_size);

    Float_data float_data{std::vector<float>(extended_input_data.data.size() * 2, 0.0f), extended_input_data.shape};

    memcpy(float_data.data.data(), extended_input_data.data.data(), extended_input_data.data.size() * 2 * sizeof(float));
    for (const auto a : outer_fft_axes) {
        // Perform IDFT with respect to the current outer axis, a.
        std::cout << "current outer fft axes: " << a << "\n";
        float_data = calculate_idft_with_respect_to_given_axis(float_data, a, output_ifft_shape[a]);
    }
    std::cout << "float_data.shape: " << float_data.shape << "\n";
    std::cout << "float_data.data: [";
    for (const auto x : float_data.data) {
        std::cout << x << ", ";
    }
    std::cout << "]\n";

    const std::vector<int64_t> inner_axis = {last_axis};

//     Float_data float_data{input_data, input_data_shape};
//     for (const auto a : outer_fft_axes) {
//         // Perform IDFT with respect to the current outer axis, a.
//         std::cout << "current outer fft axes: " << a << "\n";
//         float_data = calculate_idft_with_respect_to_given_axis(float_data, a, output_ifft_shape[a]);
//     }
//     std::cout << "float_data.shape: " << float_data.shape << "\n";
//     std::cout << "float_data.data: [";
//     for (const auto x : float_data.data) {
//         std::cout << x << ", ";
//     }
//     std::cout << "]\n";
//
//     const std::vector<int64_t> inner_axis = {last_axis};
//     const auto extended_input_data =  extend_input_data(float_data.data, float_data.shape, inner_axis, last_signal_size);
// //     const auto extended_input_data =  extend_input_data(input_data, input_data_shape, axes_data, last_signal_size);

    std::vector<complex_type> ifft_result(shape_size(output_ifft_shape) / 2);

    // Here calculation of IDFT
    fft(float_data.data.data(),
        float_data.shape,
        inner_axis.data(),
        Shape{inner_axis.size()},
        reinterpret_cast<float*>(ifft_result.data()),
        output_ifft_shape,
        FFTKind::Inverse);
//     fft(reinterpret_cast<const float*>(extended_input_data.data.data()),
//         extended_input_data.shape,
//         inner_axis.data(),
//         Shape{inner_axis.size()},
//         reinterpret_cast<float*>(ifft_result.data()),
//         output_ifft_shape,
//         FFTKind::Inverse);
// //     fft(reinterpret_cast<const float*>(extended_input_data.data.data()),
// //         extended_input_data.shape,
// //         axes_data.data(),
// //         Shape{axes_data.size()},
// //         reinterpret_cast<float*>(ifft_result.data()),
// //         output_ifft_shape,
// //         FFTKind::Inverse);

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