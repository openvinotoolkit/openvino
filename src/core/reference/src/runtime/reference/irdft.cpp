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

// Calculation of IRDFT
void irdft_calculation(const float* input_data,
                       const Shape& input_data_shape,
                       const std::vector<int64_t>& axes_data,
                       float* fft_result,
                       const Shape& fft_output_shape) {
    std::cout << "input_data_shape: " << input_data_shape << "\n";
    std::cout << "fft_output_shape: " << fft_output_shape << "\n";

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

    const auto reversed_output_shape = fft_common::reverse_shape_of_emulated_complex_tensor(fft_output_shape);
    std::cout << "reversed_output_shape: ";
    for (const auto d : reversed_output_shape) {
        std::cout << d << " ";
    }
    std::cout << "\n";

    const int64_t complex_data_rank = static_cast<int64_t>(input_data_shape.size()) - 1;
    const auto reversed_fft_axes = reverse_fft_axes(axes_data, complex_data_rank);
    std::cout << "complex_data_rank: " << complex_data_rank << "\n";
    std::cout << "reversed_fft_axes: ";
    for (const auto a : reversed_fft_axes) {
        std::cout << a << " ";
    }
    std::cout << "\n";

    const int64_t fft_rank = reversed_fft_axes.size();
    std::cout << "fft_rank: " << fft_rank << "\n";

    const auto fft_lengths = get_lengths(reversed_output_shape, fft_axes);
    std::cout << "fft_lengths: ";
    for (const auto a : fft_lengths) {
        std::cout << a << " ";
    }
    std::cout << "\n";

    const auto fft_strides = fft_common::compute_strides(fft_lengths);
    const int64_t fft_size = fft_strides[fft_rank];
    std::cout << "fft_strides: ";
    for (const auto a : fft_strides) {
        std::cout << a << " ";
    }
    std::cout << "\n";

    std::cout << "fft_size: " << fft_size << "\n";
    const auto outer_axes = get_outer_axes(fft_axes, complex_data_rank);
    std::cout << "outer_axes: ";
    for (const auto a : outer_axes) {
        std::cout << a << " ";
    }
    std::cout << "\n";
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
