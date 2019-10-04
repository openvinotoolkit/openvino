/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Element-wise linear quantization of floating point input values into a descrete set of floating point values.
/// @details In general there are four values that specify quantization for each element:
/// input_low, input_high, output_low, output_high.
/// Values input_low and input_high specifies the input range of quantization.
/// All input values, that are outside this range, clipped to the range before actual quantization.
/// Values output_low and output_high define minimum and maximum quantized values at the output.
struct quantize : public primitive_base<quantize> {
    CLDNN_DECLARE_PRIMITIVE(quantize)

    quantize(const primitive_id& id,
             const primitive_id& input,
             const primitive_id& input_low,
             const primitive_id& input_high,
             const primitive_id& output_low,
             const primitive_id& output_high,
             const int levels,
             const padding& output_padding = padding())
        : primitive_base(id, {input, input_low, input_high, output_low, output_high}, output_padding), levels(levels) {}

    /// @brief levels The number of quantization levels.
    int levels;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
