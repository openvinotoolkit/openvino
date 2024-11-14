// Copyright (C) 2018-2024 Intel Corporation
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
             const data_types output_data_type)
        : primitive_base(id, {input, input_low, input_high, output_low, output_high}, 1, {optional_data_type{output_data_type}})
        , levels(levels) {}

    quantize(const primitive_id& id,
             const std::vector<input_info>& inputs,
             const int levels,
             const data_types output_data_type)
        : primitive_base(id, inputs, 1, {optional_data_type{output_data_type}})
        , levels(levels) {}

    quantize() : primitive_base("", {}), levels(0) {}

    /// @brief levels The number of quantization levels.
    int levels;

    bool scale_shift_opt = false;
    bool need_post_scale = false;
    bool need_post_shift = false;
    bool need_pre_shift = false;
    bool need_clamp = false;
    bool need_min_clamp = false;
    bool need_max_clamp = false;

    bool per_tensor_input_range = false;
    bool per_tensor_input_scale = false;
    bool per_tensor_input_shift = false;
    bool per_tensor_output_range = false;
    bool per_tensor_output_scale = false;
    bool per_tensor_output_shift = false;

    float in_lo = 0.0f;
    float in_hi = 0.0f;
    float in_scale = 0.0f;
    float in_shift = 0.0f;
    float out_lo = 0.0f;
    float out_hi = 0.0f;
    float out_scale = 0.0f;
    float out_shift = 0.0f;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = cldnn::hash_combine(seed, levels);
        seed = cldnn::hash_combine(seed, scale_shift_opt);
        seed = cldnn::hash_combine(seed, need_post_scale);
        seed = cldnn::hash_combine(seed, need_post_shift);
        seed = cldnn::hash_combine(seed, need_pre_shift);
        seed = cldnn::hash_combine(seed, need_clamp);
        seed = cldnn::hash_combine(seed, need_min_clamp);
        seed = cldnn::hash_combine(seed, need_max_clamp);
        seed = cldnn::hash_combine(seed, per_tensor_input_range);
        seed = cldnn::hash_combine(seed, per_tensor_input_scale);
        seed = cldnn::hash_combine(seed, per_tensor_input_shift);
        seed = cldnn::hash_combine(seed, per_tensor_output_range);
        seed = cldnn::hash_combine(seed, per_tensor_output_scale);
        seed = cldnn::hash_combine(seed, per_tensor_output_shift);
        seed = cldnn::hash_combine(seed, in_lo);
        seed = cldnn::hash_combine(seed, in_hi);
        seed = cldnn::hash_combine(seed, in_scale);
        seed = cldnn::hash_combine(seed, in_shift);
        seed = cldnn::hash_combine(seed, out_lo);
        seed = cldnn::hash_combine(seed, out_hi);
        seed = cldnn::hash_combine(seed, out_scale);
        seed = cldnn::hash_combine(seed, out_shift);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const quantize>(rhs);

        return levels == rhs_casted.levels &&
               scale_shift_opt == rhs_casted.scale_shift_opt &&
               need_post_scale == rhs_casted.need_post_scale &&
               need_post_shift == rhs_casted.need_post_shift &&
               need_pre_shift == rhs_casted.need_pre_shift &&
               need_clamp == rhs_casted.need_clamp &&
               need_min_clamp == rhs_casted.need_min_clamp &&
               need_max_clamp == rhs_casted.need_max_clamp &&
               per_tensor_input_range == rhs_casted.per_tensor_input_range &&
               per_tensor_input_scale == rhs_casted.per_tensor_input_scale &&
               per_tensor_input_shift == rhs_casted.per_tensor_input_shift &&
               per_tensor_output_range == rhs_casted.per_tensor_output_range &&
               per_tensor_output_scale == rhs_casted.per_tensor_output_scale &&
               per_tensor_output_shift == rhs_casted.per_tensor_output_shift &&
               in_lo == rhs_casted.in_lo &&
               in_hi == rhs_casted.in_hi &&
               in_scale == rhs_casted.in_scale &&
               in_shift == rhs_casted.in_shift &&
               out_lo == rhs_casted.out_lo &&
               out_hi == rhs_casted.out_hi &&
               out_scale == rhs_casted.out_scale &&
               out_shift == rhs_casted.out_shift;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<quantize>::save(ob);
        ob << levels;
        ob << scale_shift_opt;
        ob << need_post_scale;
        ob << need_post_shift;
        ob << need_pre_shift;
        ob << need_clamp;
        ob << need_min_clamp;
        ob << need_max_clamp;
        ob << per_tensor_input_range;
        ob << per_tensor_input_scale;
        ob << per_tensor_input_shift;
        ob << per_tensor_output_range;
        ob << per_tensor_output_scale;
        ob << per_tensor_output_shift;
        ob << in_lo;
        ob << in_hi;
        ob << in_scale;
        ob << in_shift;
        ob << out_lo;
        ob << out_hi;
        ob << out_scale;
        ob << out_shift;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<quantize>::load(ib);
        ib >> levels;
        ib >> scale_shift_opt;
        ib >> need_post_scale;
        ib >> need_post_shift;
        ib >> need_pre_shift;
        ib >> need_clamp;
        ib >> need_min_clamp;
        ib >> need_max_clamp;
        ib >> per_tensor_input_range;
        ib >> per_tensor_input_scale;
        ib >> per_tensor_input_shift;
        ib >> per_tensor_output_range;
        ib >> per_tensor_output_scale;
        ib >> per_tensor_output_shift;
        ib >> in_lo;
        ib >> in_hi;
        ib >> in_scale;
        ib >> in_shift;
        ib >> out_lo;
        ib >> out_hi;
        ib >> out_scale;
        ib >> out_shift;
    }
};
}  // namespace cldnn
