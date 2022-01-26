// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
             const data_types output_data_type,
             const primitive_id& ext_prim_id = "",
             const padding& output_padding = padding())
        : primitive_base(id, {input, input_low, input_high, output_low, output_high}, ext_prim_id, output_padding, optional_data_type{output_data_type})
        , levels(levels) {}

    /// @brief levels The number of quantization levels.
    int levels;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
