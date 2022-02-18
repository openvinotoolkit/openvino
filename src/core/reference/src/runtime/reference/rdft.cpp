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

void clip_and_write_result(const std::vector<int64_t>& axes_data,
                           const std::vector<float>& fft_result,
                           const Shape& output_fft_shape,
                           float* rdft_result)
{
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