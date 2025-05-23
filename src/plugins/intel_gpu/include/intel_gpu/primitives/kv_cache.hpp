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

    using QuantizationAttributes = ov::op::internal::DynamicQuantize::Attributes;

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
    QuantizationAttributes quantization_attributes;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, concat_axis);
        seed = hash_combine(seed, gather_axis);
        seed = hash_combine(seed, indirect);
        seed = hash_combine(seed, compressed);
        seed = hash_range(seed, quantization_attributes.scales_zp_output_order.begin(), quantization_attributes.scales_zp_output_order.end());
        seed = hash_range(seed, quantization_attributes.group_sizes.begin(), quantization_attributes.group_sizes.end());
        seed = hash_combine(seed, quantization_attributes.quantization_type);
        seed = hash_combine(seed, quantization_attributes.quantization_dt.hash());
        seed = hash_combine(seed, quantization_attributes.scale_dt.hash());
        seed = hash_combine(seed, quantization_attributes.zp_dt.hash());
        seed = hash_combine(seed, quantization_attributes.output_storage_type);;

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
               quantization_attributes.scales_zp_output_order == rhs_casted.quantization_attributes.scales_zp_output_order &&
               quantization_attributes.output_storage_type == rhs_casted.quantization_attributes.output_storage_type &&
               quantization_attributes.group_sizes == rhs_casted.quantization_attributes.group_sizes &&
               quantization_attributes.quantization_dt == rhs_casted.quantization_attributes.quantization_dt &&
               quantization_attributes.scale_dt == rhs_casted.quantization_attributes.scale_dt &&
               quantization_attributes.zp_dt == rhs_casted.quantization_attributes.zp_dt &&
               quantization_attributes.quantization_type == rhs_casted.quantization_attributes.quantization_type;
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
        ob << make_data(&quantization_attributes.quantization_type, sizeof(quantization_attributes.quantization_type));
        ob << make_data(&quantization_attributes.quantization_dt, sizeof(quantization_attributes.quantization_dt));
        ob << make_data(&quantization_attributes.scale_dt, sizeof(quantization_attributes.scale_dt));
        ob << make_data(&quantization_attributes.zp_dt, sizeof(quantization_attributes.zp_dt));
        ob << make_data(&quantization_attributes.output_storage_type, sizeof(quantization_attributes.output_storage_type));
        ob << quantization_attributes.scales_zp_output_order;
        ob << quantization_attributes.group_sizes;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<kv_cache>::load(ib);
        ov::PartialShape data_shape;
        ov::element::Type_t data_type = ov::element::Type_t::dynamic;
        std::string variable_id;
        ib >> variable_id;
        ib >> data_shape;
        ib >> make_data(&data_type, sizeof(ov::element::Type_t));
        variable_info = { data_shape, data_type, variable_id };
        ib >> concat_axis;
        ib >> gather_axis;
        ib >> indirect;
        ib >> compressed;
        ib >> make_data(&quantization_attributes.quantization_type, sizeof(quantization_attributes.quantization_type));
        ib >> make_data(&quantization_attributes.quantization_dt, sizeof(quantization_attributes.quantization_dt));
        ib >> make_data(&quantization_attributes.scale_dt, sizeof(quantization_attributes.scale_dt));
        ib >> make_data(&quantization_attributes.zp_dt, sizeof(quantization_attributes.zp_dt));
        ib >> make_data(&quantization_attributes.output_storage_type, sizeof(quantization_attributes.output_storage_type));
        ib >> quantization_attributes.scales_zp_output_order;
        ib >> quantization_attributes.group_sizes;
    }

    size_t get_compression_scales_inputs_num() const {
        if (compressed) {
            return 1;
        } else {
            return 0;
        }
    }

    size_t get_compression_zp_inputs_num() const {
        if (compressed &&
            quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
            quantization_attributes.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
            return 1;
        } else {
            return 0;
        }
    }
};
}  // namespace cldnn
