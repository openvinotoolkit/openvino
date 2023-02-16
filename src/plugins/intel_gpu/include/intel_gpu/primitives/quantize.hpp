// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Element-wise linear quantization of floating point input values into a descrete set of floating point values.
/// @details In general there are four values that specify quantization for each element:
/// input_low, input_high, output_low, output_high.
/// Values input_low and input_high specifies the input range of quantization.
/// All input values, that are outside this range, clipped to the range before actual quantization.
/// Values output_low and output_high define minimum and maximum quantized values at the output.
struct quantize : public primitive_base<quantize> {
    CLDNN_DECLARE_PRIMITIVE(quantize)

    quantize(const primitive_id& id,
             const input_info& input,
             const input_info& input_low,
             const input_info& input_high,
             const input_info& output_low,
             const input_info& output_high,
             const int levels,
             const data_types output_data_type,
             const padding& output_padding = padding())
        : primitive_base(id, {input, input_low, input_high, output_low, output_high}, {output_padding}, {optional_data_type{output_data_type}})
        , levels(levels) {}

    /// @brief levels The number of quantization levels.
    int levels;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = cldnn::hash_combine(seed, levels);
        return seed;
    }
};
}  // namespace cldnn
