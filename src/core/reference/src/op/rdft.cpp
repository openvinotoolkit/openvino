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

#include "openvino/reference/rdft.hpp"

#include <complex>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/fft.hpp"
#include "openvino/reference/utils/fft_common.hpp"

namespace ov {
namespace reference {
namespace {
using complex_type = std::complex<float>;

// This function clips transformed axes and writes the result into output
void clip_and_write_result(const std::vector<int64_t>& axes_data,
                           const std::vector<float>& fft_result,
                           const Shape& output_fft_shape,
                           float* rdft_result) {
    auto rdft_result_shape = output_fft_shape;
    const auto last_axis = axes_data.back();
    rdft_result_shape[last_axis] = rdft_result_shape[last_axis] / 2 + 1;

    const auto reversed_rdft_result_shape = fft_common::reverse_shape_of_emulated_complex_tensor(rdft_result_shape);
    const auto rdft_output_strides = fft_common::compute_strides(reversed_rdft_result_shape);
    const auto reversed_output_fft_shape = fft_common::reverse_shape_of_emulated_complex_tensor(output_fft_shape);
    const auto output_fft_strides = fft_common::compute_strides(reversed_output_fft_shape);
    const auto rdft_output_size = rdft_output_strides.back();

    complex_type* complex_output_ptr = reinterpret_cast<complex_type*>(rdft_result);
    const complex_type* complex_input_ptr = reinterpret_cast<const complex_type*>(fft_result.data());
    for (int64_t i = 0; i < rdft_output_size; ++i) {
        const auto coords = fft_common::coords_from_index(i, rdft_output_strides);
        const int64_t input_offset = fft_common::offset_from_coords_and_strides(coords, output_fft_strides);
        complex_output_ptr[i] = complex_input_ptr[input_offset];
    }
}
}  // namespace

void rdft(const std::vector<float>& input_data,
          const Shape& input_data_shape,
          const std::vector<int64_t>& axes_data,
          const Shape& output_fft_shape,
          float* rdft_result) {
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
}  // namespace ov
