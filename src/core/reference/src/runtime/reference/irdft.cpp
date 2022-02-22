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

// To simplify calculation of strides for all axes of 'shape' of some complex
// tensor, we reverse numbers in 'shape'. Because we have no native support for
// complex numbers in tensors, we interpret FFT input tensors of the shape
// [N_0, ..., N_{r - 1}, 2] as a complex tensor with the shape
// [N_0, ..., N_{r - 1}]. Hence, we convert 'shape=[N_0, ..., N_{r - 1}, 2]'
// into [N_{r - 1}, ..., N_0].
std::vector<int64_t> reverse_shape(const Shape& shape) {
    size_t complex_data_rank = shape.size() - 1;

    std::vector<int64_t> reversed_shape(complex_data_rank);
    for (size_t i = 0; i < complex_data_rank; ++i) {
        reversed_shape[i] = static_cast<int64_t>(shape[complex_data_rank - i - 1]);
    }
    return reversed_shape;
}

// Calculates strides for all axes.
std::vector<int64_t> compute_strides(const std::vector<int64_t>& v) {
    std::vector<int64_t> strides(v.size() + 1);
    int64_t stride = 1;
    for (size_t i = 0; i < v.size(); ++i) {
        strides[i] = stride;
        stride *= v[i];
    }
    strides.back() = stride;
    return strides;
}

// When we reverted shape, we need to revert FFT axes.
void reverse_fft_axes(std::vector<int64_t>& axes, int64_t complex_data_rank) {
    for (int64_t& axis : axes) {
        axis = complex_data_rank - 1 - axis;
    }
}

// Calculating coordinates c_0, ..., c_{k - 1} from the index of the form
// c_0 * strides[0] + ... c_{k - 1} * strides[k - 1]
// where k is the number of strides.
std::vector<int64_t> coords_from_index(int64_t index, const std::vector<int64_t>& strides) {
    int64_t num_of_axes = static_cast<int64_t>(strides.size()) - 1;
    if (num_of_axes == 0) {
        return std::vector<int64_t>{};
    }
    std::vector<int64_t> coords(num_of_axes);
    int64_t curr = index;
    for (int64_t j = num_of_axes - 1; j >= 1; --j) {
        coords[j] = curr / strides[j];
        curr %= strides[j];
    }
    coords[0] = curr;
    return coords;
}
}

void irdft(const std::vector<float>& input_data,
           const Shape& input_data_shape,
           const std::vector<int64_t>& axes_data,
           const Shape& output_shape,
           float* rdft_result) {
    auto shape_of_extended_input_data = input_data_shape;
    for (const auto axis : axes_data) {
        shape_of_extended_input_data[axis] = 2 * (input_data_shape[axis] - 1);
    }

    const auto reversed_input_data_shape = reverse_shape(input_data_shape);
    const auto reversed_extended_input_data_shape = reverse_shape(shape_of_extended_input_data);

    const auto input_data_strides = compute_strides(reversed_input_data_shape);
    const auto extended_input_data_strides = compute_strides(reversed_extended_input_data_shape);

    const int64_t extended_input_data_size = shape_size(reversed_extended_input_data_shape);
    std::vector<complex_type> extended_input_data(extended_input_data_size);

//    const complex_type* complex_input_data_ptr = reinterpret_cast<const complex_type*>(input_data.data());
//    complex_type* extended_data_ptr = extended_input_data.data();

    for (int64_t i = 0; i < extended_input_data_size; ++i) {
        const auto coords = coords_from_index(i, extended_input_data_strides);
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph