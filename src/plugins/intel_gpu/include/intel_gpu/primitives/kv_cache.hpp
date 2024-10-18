// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/variable.hpp"
#include "ov_ops/dynamic_quantize.hpp"

#include <vector>

namespace cldnn {

struct kv_cache : public primitive_base<kv_cache> {
    CLDNN_DECLARE_PRIMITIVE(kv_cache)

    using QuantizationConfig = ov::op::internal::QuantizationConfig;

    kv_cache() : primitive_base("", {}) {}

    kv_cache(const primitive_id& id,
             const std::vector<input_info>& inputs,
             const ov::op::util::VariableInfo& variable_info,
             const int64_t concat_axis,
             const int64_t gather_axis,
             const bool indirect)
        : primitive_base(id, inputs)
        , variable_info(variable_info)
        , concat_axis(concat_axis)
        , gather_axis(gather_axis)
        , indirect(indirect) {}

    ov::op::util::VariableInfo variable_info;
    int64_t concat_axis = 0;
    int64_t gather_axis = 0;
    bool indirect = false;

    bool compressed = false;
    bool combine_scales_and_zp = false;
    QuantizationConfig quantization_config;
    std::vector<uint64_t> scales_zp_output_order = {};

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, concat_axis);
        seed = hash_combine(seed, gather_axis);
        seed = hash_combine(seed, indirect);
        seed = hash_combine(seed, compressed);
        seed = hash_combine(seed, combine_scales_and_zp);
        seed = hash_range(seed, scales_zp_output_order.begin(), scales_zp_output_order.end());
        seed = hash_range(seed, quantization_config.group_sizes.begin(), quantization_config.group_sizes.end());
        seed = hash_combine(seed, quantization_config.mode);
        seed = hash_combine(seed, quantization_config.quantization_dt.hash());
        seed = hash_combine(seed, quantization_config.scale_dt.hash());
        seed = hash_combine(seed, quantization_config.zp_dt.hash());

        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const kv_cache>(rhs);

        return variable_info == rhs_casted.variable_info &&
               concat_axis == rhs_casted.concat_axis &&
               gather_axis == rhs_casted.gather_axis &&
               indirect == rhs_casted.indirect &&
               compressed == rhs_casted.compressed &&
               scales_zp_output_order == rhs_casted.scales_zp_output_order &&
               combine_scales_and_zp == rhs_casted.combine_scales_and_zp &&
               quantization_config == rhs_casted.quantization_config;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<kv_cache>::save(ob);
        ov::element::Type_t data_type = variable_info.data_type;
        ob << variable_info.variable_id;
        ob << variable_info.data_shape;
        ob << make_data(&data_type, sizeof(ov::element::Type_t));
        ob << concat_axis;
        ob << gather_axis;
        ob << indirect;
        ob << compressed;
        ob << combine_scales_and_zp;
        ob << scales_zp_output_order;
        ob << quantization_config.group_sizes;
        ob << make_data(&quantization_config.mode, sizeof(quantization_config.mode));
        ob << make_data(&quantization_config.quantization_dt, sizeof(quantization_config.quantization_dt));
        ob << make_data(&quantization_config.scale_dt, sizeof(quantization_config.scale_dt));
        ob << make_data(&quantization_config.zp_dt, sizeof(quantization_config.zp_dt));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<kv_cache>::load(ib);
        ov::PartialShape data_shape;
        ov::element::Type_t data_type = ov::element::Type_t::undefined;
        std::string variable_id;
        ib >> variable_id;
        ib >> data_shape;
        ib >> make_data(&data_type, sizeof(ov::element::Type_t));
        variable_info = { data_shape, data_type, variable_id };
        ib >> concat_axis;
        ib >> gather_axis;
        ib >> indirect;
        ib >> compressed;
        ib >> combine_scales_and_zp;
        ib >> scales_zp_output_order;
        ib >> quantization_config.group_sizes;
        ib >> make_data(&quantization_config.mode, sizeof(quantization_config.mode));
        ib >> make_data(&quantization_config.quantization_dt, sizeof(quantization_config.quantization_dt));
        ib >> make_data(&quantization_config.scale_dt, sizeof(quantization_config.scale_dt));
        ib >> make_data(&quantization_config.zp_dt, sizeof(quantization_config.zp_dt));
    }

    size_t get_compression_scales_inputs_num() const {
        if (compressed) {
            return 1;
        } else {
            return 0;
        }
    }

    size_t get_compression_zp_inputs_num() const {
        if (compressed && quantization_config.is_asymmetric_quantization() && !combine_scales_and_zp) {
            return 1;
        } else {
            return 0;
        }
    }
};
}  // namespace cldnn
