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
}  // namespace

// Calculation of IRDFT
void irdft(const float* input_data,
           const Shape& input_data_shape,
           const std::vector<int64_t>& axes_data,
           float* fft_result,
           const Shape& output_shape) {
    std::cout << "input_data_shape: " << input_data_shape << "\n";
    std::cout << "output_shape:     " << output_shape << "\n";

    const complex_type* complex_input_data_ptr = reinterpret_cast<const complex_type*>(input_data);
    complex_type* complex_output_ptr = reinterpret_cast<complex_type*>(fft_result);
    std::cout << "input_data pointer:  " << complex_input_data_ptr << "\n";
    std::cout << "output data pointer: " << complex_output_ptr << "\n";

    const auto fft_axes = axes_data;
    std::cout << "fft_axes: ";
    for (const auto a : fft_axes) {
        std::cout << a << " ";
    }
    std::cout << "\n";

    const auto reversed_output_shape = fft_common::reverse_shape_of_emulated_complex_tensor(output_shape);
    std::cout << "reversed_output_shape: ";
    for (const auto d : reversed_output_shape) {
        std::cout << d << " ";
    }
    std::cout << "\n";

    const int64_t complex_data_rank = static_cast<int64_t>(input_data_shape.size()) - 1;
    const auto reversed_axes = reverse_fft_axes(axes_data, complex_data_rank);
    std::cout << "complex_data_rank: " << complex_data_rank << "\n";
    std::cout << "reversed_axes: ";
    for (const auto a : reversed_axes) {
        std::cout << a << " ";
    }
    std::cout << "\n";

    const int64_t fft_rank = reversed_axes.size();
    std::cout << "fft_rank: " << fft_rank << "\n";
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
