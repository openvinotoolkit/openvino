//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/runtime/reference/interpolate.hpp"

using namespace ngraph::runtime::reference;

float InterpolateEvalHelper::triangle_coeff(float dz)
{
    return std::max(0.0f, 1.0f - std::fabs(dz));
}

std::array<float, 4> InterpolateEvalHelper::get_cubic_coeff(float s, float a)
{
    std::array<float, 4> coeff;
    float abs_s = std::fabs(s);
    coeff[0] = static_cast<float>(
        ((a * (abs_s + 1) - 5 * a) * (abs_s + 1) + 8 * a) * (abs_s + 1) - 4 * a);
    coeff[1] = static_cast<float>(((a + 2) * abs_s - (a + 3)) * abs_s * abs_s + 1);
    coeff[2] = static_cast<float>(
        ((a + 2) * (1 - abs_s) - (a + 3)) * (1 - abs_s) * (1 - abs_s) + 1);
    coeff[3] = static_cast<float>(
        ((a * (2 - abs_s) - 5 * a) * (2 - abs_s) + 8 * a) * (2 - abs_s) - 4 * a);
    return coeff;
}
