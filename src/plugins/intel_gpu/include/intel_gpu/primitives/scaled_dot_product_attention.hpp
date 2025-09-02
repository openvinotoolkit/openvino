// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

#include "ov_ops/dynamic_quantize.hpp"

namespace cldnn {

struct scaled_dot_product_attention : public primitive_base<scaled_dot_product_attention> {
    CLDNN_DECLARE_PRIMITIVE(scaled_dot_product_attention)
    enum ScaledDotProductAttentionInputIdx {
        QUERY = 0,
        KEY = 1,
        VALUE = 2,
        ATTN_MASK = 3,
        SCALE = 4,
        SINK = 5,
        KEY_SCALES = 6,
        KEY_ZP = 7,
        VALUE_SCALES = 8,
        VALUE_ZP = 9
    };

    using QuantizationAttributes = ov::op::internal::DynamicQuantize::Attributes;

    scaled_dot_product_attention() : primitive_base("", {}) {}

    /// @brief Constructs scaled_dot_product_attention primitive.
    /// @param id This primitive id.
    /// @param inputs Input data primitives id (query, keys, values, [attention_mask], [scale], [sink], [keys scales], [keys zp], [values scales], [values zp]).
    /// @param is_causal If true, assumes causal attention masking. In this case attention_mask input is ignored.
    scaled_dot_product_attention(const primitive_id& id,
                                 const std::vector<cldnn::input_info> inputs,
                                 bool is_causal,
                                 int64_t indirect_axis = -1,
                                 const std::vector<int64_t>& input_q_transpose_order = {},
                                 const std::vector<int64_t>& input_k_transpose_order = {},
                                 const std::vector<int64_t>& input_v_transpose_order = {},
                                 const std::vector<int64_t>& output_transpose_order = {},
                                 const QuantizationAttributes& quantization_attributes = {},
                                 bool is_kv_compressed = false)
        : primitive_base(id, inputs)
        , is_causal(is_causal)
        , indirect_axis(indirect_axis)
        , is_kv_compressed(is_kv_compressed)
        , quantization_attributes(quantization_attributes)
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

                if (quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
                    quantization_attributes.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar)
                    data_inputs_num -= 2; // zp
            }
            has_attn_mask_input = data_inputs_num > ScaledDotProductAttentionInputIdx::ATTN_MASK;
            has_scale_input = data_inputs_num > ScaledDotProductAttentionInputIdx::SCALE;
            has_sink_input = data_inputs_num > ScaledDotProductAttentionInputIdx::SINK;
        }

    bool is_causal = false;
    bool has_attn_mask_input = false;
    bool has_scale_input = false;
    bool has_sink_input = false;
    int64_t indirect_axis = -1;

    bool is_kv_compressed = false;
    QuantizationAttributes quantization_attributes;

    std::vector<int64_t> input_q_transpose_order;
    std::vector<int64_t> input_k_transpose_order;
    std::vector<int64_t> input_v_transpose_order;
    std::vector<int64_t> output_transpose_order;

    std::optional<float> attn_mask_val{};
    std::optional<float> scale_val{};

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, is_causal);
        seed = hash_combine(seed, has_attn_mask_input);
        seed = hash_combine(seed, has_scale_input);
        seed = hash_combine(seed, has_sink_input);
        seed = hash_combine(seed, indirect_axis);
        seed = hash_range(seed, input_q_transpose_order.begin(), input_q_transpose_order.end());
        seed = hash_range(seed, input_k_transpose_order.begin(), input_k_transpose_order.end());
        seed = hash_range(seed, input_v_transpose_order.begin(), input_v_transpose_order.end());
        seed = hash_range(seed, output_transpose_order.begin(), output_transpose_order.end());
        seed = hash_combine(seed, attn_mask_val.has_value());
        if (attn_mask_val) {
            seed = hash_combine(seed, attn_mask_val.value());
        }
        seed = hash_combine(seed, scale_val.has_value());
        if (scale_val) {
            seed = hash_combine(seed, scale_val.value());
        }
        seed = hash_combine(seed, is_kv_compressed);
        seed = hash_range(seed, quantization_attributes.scales_zp_output_order.begin(), quantization_attributes.scales_zp_output_order.end());
        seed = hash_range(seed, quantization_attributes.group_sizes.begin(), quantization_attributes.group_sizes.end());
        seed = hash_combine(seed, quantization_attributes.quantization_type);
        seed = hash_combine(seed, quantization_attributes.quantization_dt.hash());
        seed = hash_combine(seed, quantization_attributes.scale_dt.hash());
        seed = hash_combine(seed, quantization_attributes.zp_dt.hash());
        seed = hash_combine(seed, quantization_attributes.output_storage_type);

        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const scaled_dot_product_attention>(rhs);

        return is_causal == rhs_casted.is_causal &&
               has_attn_mask_input == rhs_casted.has_attn_mask_input &&
               has_scale_input == rhs_casted.has_scale_input &&
               has_sink_input == rhs_casted.has_sink_input &&
               indirect_axis == rhs_casted.indirect_axis &&
               input_q_transpose_order == rhs_casted.input_q_transpose_order &&
               input_k_transpose_order == rhs_casted.input_k_transpose_order &&
               input_v_transpose_order == rhs_casted.input_v_transpose_order &&
               output_transpose_order == rhs_casted.output_transpose_order &&
               attn_mask_val == rhs_casted.attn_mask_val &&
               scale_val == rhs_casted.scale_val &&
               is_kv_compressed == rhs_casted.is_kv_compressed &&
               quantization_attributes.scales_zp_output_order == rhs_casted.quantization_attributes.scales_zp_output_order &&
               quantization_attributes.output_storage_type == rhs_casted.quantization_attributes.output_storage_type &&
               quantization_attributes.group_sizes == rhs_casted.quantization_attributes.group_sizes &&
               quantization_attributes.quantization_dt == rhs_casted.quantization_attributes.quantization_dt &&
               quantization_attributes.scale_dt == rhs_casted.quantization_attributes.scale_dt &&
               quantization_attributes.zp_dt == rhs_casted.quantization_attributes.zp_dt &&
               quantization_attributes.quantization_type == rhs_casted.quantization_attributes.quantization_type;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<scaled_dot_product_attention>::save(ob);
        ob << is_causal;
        ob << is_kv_compressed;
        ob << has_attn_mask_input;
        ob << has_scale_input;
        ob << has_sink_input;
        ob << indirect_axis;
        ob << input_q_transpose_order;
        ob << input_k_transpose_order;
        ob << input_v_transpose_order;
        ob << output_transpose_order;
        ob << attn_mask_val.has_value();
        if (attn_mask_val) {
            ob << make_data(&attn_mask_val.value(), sizeof(attn_mask_val.value()));
        }
        ob << scale_val.has_value();
        if (scale_val) {
            ob << make_data(&scale_val.value(), sizeof(scale_val.value()));
        }
        ob << make_data(&quantization_attributes.quantization_type, sizeof(quantization_attributes.quantization_type));
        ob << make_data(&quantization_attributes.quantization_dt, sizeof(quantization_attributes.quantization_dt));
        ob << make_data(&quantization_attributes.scale_dt, sizeof(quantization_attributes.scale_dt));
        ob << make_data(&quantization_attributes.zp_dt, sizeof(quantization_attributes.zp_dt));
        ob << make_data(&quantization_attributes.output_storage_type, sizeof(quantization_attributes.output_storage_type));
        ob << quantization_attributes.scales_zp_output_order;
        ob << quantization_attributes.group_sizes;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<scaled_dot_product_attention>::load(ib);
        ib >> is_causal;
        ib >> is_kv_compressed;
        ib >> has_attn_mask_input;
        ib >> has_scale_input;
        ib >> has_sink_input;
        ib >> indirect_axis;
        ib >> input_q_transpose_order;
        ib >> input_k_transpose_order;
        ib >> input_v_transpose_order;
        ib >> output_transpose_order;
        bool has_attn_mask_val;
        ib >> has_attn_mask_val;
        if (has_attn_mask_val) {
            ib >> make_data(&attn_mask_val.emplace(), sizeof(attn_mask_val.value()));
        }
        bool has_scale_val;
        ib >> has_scale_val;
        if (has_scale_val) {
            ib >> make_data(&scale_val.emplace(), sizeof(scale_val.value()));
        }
        ib >> make_data(&quantization_attributes.quantization_type, sizeof(quantization_attributes.quantization_type));
        ib >> make_data(&quantization_attributes.quantization_dt, sizeof(quantization_attributes.quantization_dt));
        ib >> make_data(&quantization_attributes.scale_dt, sizeof(quantization_attributes.scale_dt));
        ib >> make_data(&quantization_attributes.zp_dt, sizeof(quantization_attributes.zp_dt));
        ib >> make_data(&quantization_attributes.output_storage_type, sizeof(quantization_attributes.output_storage_type));
        ib >> quantization_attributes.scales_zp_output_order;
        ib >> quantization_attributes.group_sizes;
    }

    size_t get_compression_scales_inputs_num() const {
        if (is_kv_compressed) {
            return 2;
        } else {
            return 0;
        }
    }

    size_t get_compression_zp_inputs_num() const {
        if (is_kv_compressed &&
            quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
            quantization_attributes.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
            return 2;
        } else {
            return 0;
        }
    }
};
}  // namespace cldnn
