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

#include "ngraph/runtime/reference/rdft.hpp"

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

// Calculates offset of value using corresponding coordinates and strides.
int64_t offset_from_coords_and_strides(const std::vector<int64_t>& coords, const std::vector<int64_t>& strides) {
    int64_t offset = 0;
    int64_t num_of_axes = coords.size();
    for (int64_t i = 0; i < num_of_axes; ++i) {
        offset += coords[i] * strides[i];
    }
    return offset;
}

// This function clips transformed axes and writes the result into output
void clip_and_write_result(const std::vector<int64_t>& axes_data,
                           const std::vector<float>& fft_result,
                           const Shape& output_fft_shape,
                           float* rdft_result)
{
    auto rdft_result_shape = output_fft_shape;
    for (const auto axis : axes_data) {
        rdft_result_shape[axis] = rdft_result_shape[axis] / 2 + 1;
    }

    const auto reversed_rdft_result_shape = reverse_shape(rdft_result_shape);
    const auto rdft_output_strides = compute_strides(reversed_rdft_result_shape);

    const auto reversed_output_fft_shape = reverse_shape(output_fft_shape);
    const auto output_fft_strides = compute_strides(reversed_output_fft_shape);
    const auto rdft_output_size = rdft_output_strides.back();

    complex_type* complex_output_ptr = reinterpret_cast<complex_type*>(rdft_result);
    const complex* complex_input_ptr = reinterpret_cast<const complex_type*>(fft_result.data());
    for (int64_t i = 0; i < rdft_output_size; ++i) {
        const auto coords = coords_from_index(i, rdft_output_strides);
        const int64_t input_offset = offset_from_coords_and_strides(coords, output_fft_strides);
        complex_output_ptr[i] = complex_input_ptr[input_offset];
    }
}
}  // namespace

void rdft(const std::vector<float>& input_data,
          const Shape& input_data_shape,
          const std::vector<int64_t>& axes_data,
          const Shape& output_fft_shape,
          float* rdft_result)
{
    // Converting input data to complex type and calculation of DFT with such data.
    size_t input_data_size = input_data.size();
    std::vector<complex_type> complex_data(input_data_size);
    for (size_t i = 0; i < input_data_size; ++i) {
        complex_data[i] = complex_type{input_data[i], 0.0f};
    }

    auto input_shape_for_fft = input_data_shape;
    input_shape_for_fft.push_back(2);

    std::vector<float> fft_result(shape_size(output_fft_shape), 0.0f);

    fft(reinterpret_cast<const float*>(complex_data.data()),
        input_shape_for_fft,
        axes_data.data(),
        Shape{axes_data.size()},
        fft_result.data(),
        output_fft_shape,
        FFTKind::Forward);

    clip_and_write_result(axes_data, fft_result, output_fft_shape, rdft_result);
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph