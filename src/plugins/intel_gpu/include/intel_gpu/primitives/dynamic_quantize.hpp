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

    using QuantizationConfig = ov::op::internal::QuantizationConfig;

    dynamic_quantize() : primitive_base("", {}) {}

    /// @brief Constructs dynamic_quantize primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param group_sizes Quantization group size
    /// @param data_type Output data type of quantized
    /// @param output_size Output data size of the primitive
    dynamic_quantize(const primitive_id& id,
           const input_info& input,
           const QuantizationConfig& config,
           const bool combine_scales_and_zp = false,
           const std::vector<uint64_t>& scales_zp_output_order = {})
           : primitive_base(id, {input})
           , combine_scales_and_zp(combine_scales_and_zp)
           , quantization_config(config)
           , scales_zp_output_order(scales_zp_output_order) {
        num_outputs = 2;
        if (config.is_asymmetric_quantization() && !combine_scales_and_zp)
            num_outputs++;
    }

    bool combine_scales_and_zp = false;
    QuantizationConfig quantization_config;
    std::vector<uint64_t> scales_zp_output_order = {};

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, scales_zp_output_order.begin(), scales_zp_output_order.end());
        seed = hash_range(seed, quantization_config.group_sizes.begin(), quantization_config.group_sizes.end());
        seed = hash_combine(seed, quantization_config.type);
        seed = hash_combine(seed, quantization_config.quantization_dt.hash());
        seed = hash_combine(seed, quantization_config.scale_dt.hash());
        seed = hash_combine(seed, quantization_config.zp_dt.hash());
        seed = hash_combine(seed, combine_scales_and_zp);

        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const dynamic_quantize>(rhs);

        return scales_zp_output_order == rhs_casted.scales_zp_output_order ||
               combine_scales_and_zp == rhs_casted.combine_scales_and_zp ||
               quantization_config == rhs_casted.quantization_config;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<dynamic_quantize>::save(ob);

        ob << combine_scales_and_zp;
        ob << scales_zp_output_order;
        ob << quantization_config.group_sizes;
        ob << make_data(&quantization_config.type, sizeof(quantization_config.type));
        ob << make_data(&quantization_config.quantization_dt, sizeof(quantization_config.quantization_dt));
        ob << make_data(&quantization_config.scale_dt, sizeof(quantization_config.scale_dt));
        ob << make_data(&quantization_config.zp_dt, sizeof(quantization_config.zp_dt));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<dynamic_quantize>::load(ib);

        ib >> combine_scales_and_zp;
        ib >> scales_zp_output_order;
        ib >> quantization_config.group_sizes;
        ib >> make_data(&quantization_config.type, sizeof(quantization_config.type));
        ib >> make_data(&quantization_config.quantization_dt, sizeof(quantization_config.quantization_dt));
        ib >> make_data(&quantization_config.scale_dt, sizeof(quantization_config.scale_dt));
        ib >> make_data(&quantization_config.zp_dt, sizeof(quantization_config.zp_dt));
    }
};
}  // namespace cldnn
