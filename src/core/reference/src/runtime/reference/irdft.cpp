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

    auto shape_of_complex_data = shape_of_extended_input_data;
    shape_of_complex_data.pop_back();
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph