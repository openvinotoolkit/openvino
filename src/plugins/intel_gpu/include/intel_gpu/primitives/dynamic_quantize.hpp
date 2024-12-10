// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

#include "ov_ops/dynamic_quantize.hpp"

namespace cldnn {

/// @brief Dynamic Quantize primitive
/// @details Performs dynamic quantization
struct dynamic_quantize : public primitive_base<dynamic_quantize> {
    CLDNN_DECLARE_PRIMITIVE(dynamic_quantize);

    using Attributes = ov::op::internal::DynamicQuantize::Attributes;

    dynamic_quantize() : primitive_base("", {}) {}

    /// @brief Constructs dynamic_quantize primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param group_sizes Quantization group size
    /// @param data_type Output data type of quantized
    /// @param output_size Output data size of the primitive
    dynamic_quantize(const primitive_id& id,
           const input_info& input,
           const Attributes& attrs,
           const size_t input_size = 3)
           : primitive_base(id, {input})
           , attrs(attrs)
           , input_size(input_size) {
        num_outputs = 2;
        if (attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
            attrs.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar)
            num_outputs++;
    }

    Attributes attrs;
    size_t input_size;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, attrs.scales_zp_output_order.begin(), attrs.scales_zp_output_order.end());
        seed = hash_range(seed, attrs.group_sizes.begin(), attrs.group_sizes.end());
        seed = hash_combine(seed, attrs.quantization_type);
        seed = hash_combine(seed, attrs.quantization_dt.hash());
        seed = hash_combine(seed, attrs.scale_dt.hash());
        seed = hash_combine(seed, attrs.zp_dt.hash());
        seed = hash_combine(seed, attrs.output_storage_type);
        seed = hash_combine(seed, input_size);

        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const dynamic_quantize>(rhs);

        return attrs.scales_zp_output_order == rhs_casted.attrs.scales_zp_output_order &&
               attrs.output_storage_type == rhs_casted.attrs.output_storage_type &&
               attrs.group_sizes == rhs_casted.attrs.group_sizes &&
               attrs.quantization_dt == rhs_casted.attrs.quantization_dt &&
               attrs.scale_dt == rhs_casted.attrs.scale_dt &&
               attrs.zp_dt == rhs_casted.attrs.zp_dt &&
               attrs.quantization_type == rhs_casted.attrs.quantization_type &&
               input_size == rhs_casted.input_size;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<dynamic_quantize>::save(ob);

        ob << make_data(&attrs.quantization_type, sizeof(attrs.quantization_type));
        ob << make_data(&attrs.quantization_dt, sizeof(attrs.quantization_dt));
        ob << make_data(&attrs.scale_dt, sizeof(attrs.scale_dt));
        ob << make_data(&attrs.zp_dt, sizeof(attrs.zp_dt));
        ob << make_data(&attrs.output_storage_type, sizeof(attrs.output_storage_type));
        ob << attrs.scales_zp_output_order;
        ob << attrs.group_sizes;
        ob << input_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<dynamic_quantize>::load(ib);

        ib >> make_data(&attrs.quantization_type, sizeof(attrs.quantization_type));
        ib >> make_data(&attrs.quantization_dt, sizeof(attrs.quantization_dt));
        ib >> make_data(&attrs.scale_dt, sizeof(attrs.scale_dt));
        ib >> make_data(&attrs.zp_dt, sizeof(attrs.zp_dt));
        ib >> make_data(&attrs.output_storage_type, sizeof(attrs.output_storage_type));
        ib >> attrs.scales_zp_output_order;
        ib >> attrs.group_sizes;
        ib >> input_size;
    }
};
}  // namespace cldnn
