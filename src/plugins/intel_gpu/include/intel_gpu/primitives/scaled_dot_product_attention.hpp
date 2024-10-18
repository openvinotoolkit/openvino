// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

#include "ov_ops/dynamic_quantize.hpp"

namespace cldnn {

struct scaled_dot_product_attention : public primitive_base<scaled_dot_product_attention> {
    CLDNN_DECLARE_PRIMITIVE(scaled_dot_product_attention)

    using QuantizationConfig = ov::op::internal::QuantizationConfig;

    scaled_dot_product_attention() : primitive_base("", {}) {}

    /// @brief Constructs scaled_dot_product_attention primitive.
    /// @param id This primitive id.
    /// @param inputs Input data primitives id (query, keys, values, [attention_mask], [scale], [keys scales], [keys zp], [values scales], [values zp]).
    /// @param is_causal If true, assumes causal attention masking. In this case attention_mask input is ignored.
    scaled_dot_product_attention(const primitive_id& id,
                                 const std::vector<cldnn::input_info> inputs,
                                 bool is_causal,
                                 int64_t indirect_axis = -1,
                                 const std::vector<int64_t>& input_q_transpose_order = {},
                                 const std::vector<int64_t>& input_k_transpose_order = {},
                                 const std::vector<int64_t>& input_v_transpose_order = {},
                                 const std::vector<int64_t>& output_transpose_order = {},
                                 bool is_kv_compressed = false,
                                 bool combine_scales_and_zp = false,
                                 const QuantizationConfig& quantization_config = {})
        : primitive_base(id, inputs)
        , is_causal(is_causal)
        , indirect_axis(indirect_axis)
        , is_kv_compressed(is_kv_compressed)
        , combine_scales_and_zp(combine_scales_and_zp)
        , quantization_config(quantization_config)
        , input_q_transpose_order(input_q_transpose_order)
        , input_k_transpose_order(input_k_transpose_order)
        , input_v_transpose_order(input_v_transpose_order)
        , output_transpose_order(output_transpose_order) {
            auto data_inputs_num = inputs.size();
            if (indirect_axis != -1) {
                data_inputs_num--;
            }
            if (is_kv_compressed) {
                data_inputs_num -= 2; // scales

                if (quantization_config.is_asymmetric_quantization() && !combine_scales_and_zp)
                    data_inputs_num -= 2; // zp
            }
            has_attn_mask_input = data_inputs_num > 3;
            has_scale_input = data_inputs_num > 4;
        }

    bool is_causal = false;
    bool has_attn_mask_input = false;
    bool has_scale_input = false;
    int64_t indirect_axis = -1;

    bool is_kv_compressed = false;
    bool combine_scales_and_zp = false;
    QuantizationConfig quantization_config;

    std::vector<int64_t> input_q_transpose_order;
    std::vector<int64_t> input_k_transpose_order;
    std::vector<int64_t> input_v_transpose_order;
    std::vector<int64_t> output_transpose_order;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, is_causal);
        seed = hash_combine(seed, has_attn_mask_input);
        seed = hash_combine(seed, has_scale_input);
        seed = hash_combine(seed, indirect_axis);
        seed = hash_range(seed, input_q_transpose_order.begin(), input_q_transpose_order.end());
        seed = hash_range(seed, input_k_transpose_order.begin(), input_k_transpose_order.end());
        seed = hash_range(seed, input_v_transpose_order.begin(), input_v_transpose_order.end());
        seed = hash_range(seed, output_transpose_order.begin(), output_transpose_order.end());
        seed = hash_combine(seed, is_kv_compressed);
        seed = hash_combine(seed, combine_scales_and_zp);
        seed = hash_range(seed, quantization_config.group_sizes.begin(), quantization_config.group_sizes.end());
        seed = hash_combine(seed, quantization_config.type);
        seed = hash_combine(seed, quantization_config.quantization_dt.hash());
        seed = hash_combine(seed, quantization_config.scale_dt.hash());
        seed = hash_combine(seed, quantization_config.zp_dt.hash());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const scaled_dot_product_attention>(rhs);

        return is_causal == rhs_casted.is_causal &&
               has_attn_mask_input == rhs_casted.has_attn_mask_input &&
               has_scale_input == rhs_casted.has_scale_input &&
               indirect_axis == rhs_casted.indirect_axis &&
               input_q_transpose_order == rhs_casted.input_q_transpose_order &&
               input_k_transpose_order == rhs_casted.input_k_transpose_order &&
               input_v_transpose_order == rhs_casted.input_v_transpose_order &&
               output_transpose_order == rhs_casted.output_transpose_order &&
               is_kv_compressed == rhs_casted.is_kv_compressed &&
               combine_scales_and_zp == rhs_casted.combine_scales_and_zp &&
               quantization_config == rhs_casted.quantization_config;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<scaled_dot_product_attention>::save(ob);
        ob << is_causal;
        ob << has_attn_mask_input;
        ob << has_scale_input;
        ob << indirect_axis;
        ob << input_q_transpose_order;
        ob << input_k_transpose_order;
        ob << input_v_transpose_order;
        ob << output_transpose_order;
        ob << is_kv_compressed;
        ob << combine_scales_and_zp;
        ob << quantization_config.group_sizes;
        ob << make_data(&quantization_config.type, sizeof(quantization_config.type));
        ob << make_data(&quantization_config.quantization_dt, sizeof(quantization_config.quantization_dt));
        ob << make_data(&quantization_config.scale_dt, sizeof(quantization_config.scale_dt));
        ob << make_data(&quantization_config.zp_dt, sizeof(quantization_config.zp_dt));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<scaled_dot_product_attention>::load(ib);
        ib >> is_causal;
        ib >> is_kv_compressed;
        ib >> has_attn_mask_input;
        ib >> has_scale_input;
        ib >> indirect_axis;
        ib >> input_q_transpose_order;
        ib >> input_k_transpose_order;
        ib >> input_v_transpose_order;
        ib >> output_transpose_order;
        ib >> combine_scales_and_zp;
        ib >> quantization_config.group_sizes;
        ib >> make_data(&quantization_config.type, sizeof(quantization_config.type));
        ib >> make_data(&quantization_config.quantization_dt, sizeof(quantization_config.quantization_dt));
        ib >> make_data(&quantization_config.scale_dt, sizeof(quantization_config.scale_dt));
        ib >> make_data(&quantization_config.zp_dt, sizeof(quantization_config.zp_dt));
    }
};
}  // namespace cldnn
