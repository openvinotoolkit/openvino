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
#include <ngraph/runtime/reference/utils/fft_common.hpp>
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
}  // namespace

void irdft(const std::vector<float>& input_data,
           const Shape& input_data_shape,
           const std::vector<int64_t>& axes_data,
           const Shape& output_ifft_shape,
           float* irdft_result) {
    auto shape_of_extended_input_data = input_data_shape;
    for (const auto axis : axes_data) {
        shape_of_extended_input_data[axis] = 2 * (input_data_shape[axis] - 1);
    }

    const auto reversed_input_data_shape = fft_common::reverse_shape(input_data_shape);
    const auto reversed_extended_input_data_shape = fft_common::reverse_shape(shape_of_extended_input_data);

    const auto input_data_strides = fft_common::compute_strides(reversed_input_data_shape);
    const auto extended_input_data_strides = fft_common::compute_strides(reversed_extended_input_data_shape);

    const int64_t extended_input_data_size = shape_size(reversed_extended_input_data_shape);
    std::vector<complex_type> extended_input_data(extended_input_data_size);

    const int64_t complex_data_rank = static_cast<int64_t>(reversed_extended_input_data_shape.size());
    const auto reversed_axes = reverse_fft_axes(axes_data, complex_data_rank);

    const complex_type* complex_input_data_ptr = reinterpret_cast<const complex_type*>(input_data.data());
    complex_type* extended_data_ptr = extended_input_data.data();

    for (int64_t i = 0; i < extended_input_data_size; ++i) {
        const auto coords = fft_common::coords_from_index(i, extended_input_data_strides);

        bool need_conj = false;
        auto coords_to_read = coords;
        for (const auto a : reversed_axes) {
            if (coords[a] >= reversed_input_data_shape[a]) {
                need_conj = true;
                coords_to_read[a] = reversed_extended_input_data_shape[a] - coords[a];
            }
        }

        const int64_t offset_to_read = fft_common::offset_from_coords_and_strides(coords_to_read, input_data_strides);

        if (need_conj) {
            extended_data_ptr[i] = std::conj(complex_input_data_ptr[offset_to_read]);
        } else {
            extended_data_ptr[i] = complex_input_data_ptr[offset_to_read];
        }
    }

    const size_t size_of_ifft_result = shape_size(output_ifft_shape);
    std::vector<complex_type> ifft_result(size_of_ifft_result);

    // Here calculation of IDFT
    fft(reinterpret_cast<const float*>(extended_input_data.data()),
        shape_of_extended_input_data,
        axes_data.data(),
        Shape{axes_data.size()},
        reinterpret_cast<float*>(ifft_result.data()),
        output_ifft_shape,
        FFTKind::Inverse);

    for (size_t i = 0; i < size_of_ifft_result; ++i) {
        irdft_result[i] = std::real(ifft_result[i]);
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph