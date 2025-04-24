// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
namespace fft_common {
// To simplify calculation of strides for all axes of 'shape' of some complex
// tensor, we reverse numbers in 'shape'. Because we have no native support for
// complex numbers in tensors, we interpret float input tensors of the shape
// [N_0, ..., N_{r - 1}, 2] as a complex tensor with the shape
// [N_0, ..., N_{r - 1}]. Hence, we convert 'shape=[N_0, ..., N_{r - 1}, 2]'
// into [N_{r - 1}, ..., N_0].
// At this time, complex tensors are supported only for FFT-like operations, as
// DFT, IDFT, RDFT
std::vector<int64_t> reverse_shape_of_emulated_complex_tensor(const Shape& shape);

// Calculates strides for all axes.
std::vector<int64_t> compute_strides(const std::vector<int64_t>& v);

// Calculating coordinates c_0, ..., c_{k - 1} from the index of the form
// c_0 * strides[0] + ... c_{k - 1} * strides[k - 1]
// where k is the number of strides.
std::vector<int64_t> coords_from_index(int64_t index, const std::vector<int64_t>& strides);

// Calculates offset of value using corresponding coordinates and strides.
int64_t offset_from_coords_and_strides(const std::vector<int64_t>& coords, const std::vector<int64_t>& strides);

// Reverse order of given axes of complex number (where last dimension represents real and imaginary part)
std::vector<int64_t> reverse_fft_axes(const std::vector<int64_t>& axes, int64_t complex_data_rank);
}  // namespace fft_common
}  // namespace reference
}  // namespace ov
