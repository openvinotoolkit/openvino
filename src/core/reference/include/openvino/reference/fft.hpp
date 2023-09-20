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

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <vector>

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace reference {
enum class FFTKind { Forward, Inverse };

void fft(const float* input_data,
         const Shape& input_data_shape,
         const int64_t* axes_data,
         const Shape& axes_data_shape,
         float* fft_result,
         const Shape& output_shape,
         FFTKind fft_kind);

void fft_postprocessing(ov::TensorVector& outputs,
                        const ov::element::Type output_type,
                        const std::vector<float>& fft_result);

std::vector<int64_t> canonicalize_axes(const int64_t* axes_data,
                                       const Shape& axes_data_shape,
                                       int64_t complex_data_rank);
}  // namespace reference
}  // namespace ov
