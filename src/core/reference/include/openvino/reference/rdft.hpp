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

#include <cstddef>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
void rdft(const std::vector<float>& input_data,
          const Shape& input_data_shape,
          const std::vector<int64_t>& axes_data,
          const Shape& output_fft_shape,
          float* rdft_result);
}  // namespace reference
}  // namespace ov
