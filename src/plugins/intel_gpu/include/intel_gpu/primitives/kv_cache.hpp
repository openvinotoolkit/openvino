// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/variable.hpp"
#include "primitive.hpp"
#include <vector>

namespace cldnn {

struct kv_cache : public primitive_base<kv_cache> {
    CLDNN_DECLARE_PRIMITIVE(kv_cache)

    kv_cache() : primitive_base("", {}) {}

    kv_cache(const primitive_id& id,
             const std::vector<input_info>& inputs,
             const ov::op::util::VariableInfo& variable_info,
             const int64_t concat_axis,
             const int64_t gather_axis,
             const bool indirect,
             const padding& output_padding = padding())
        : primitive_base(id, inputs, {output_padding})
        , variable_info(variable_info)
        , concat_axis(concat_axis)
        , gather_axis(gather_axis)
        , indirect(indirect) {}

    ov::op::util::VariableInfo variable_info;
    int64_t concat_axis = 0;
    int64_t gather_axis = 0;
    bool indirect = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, concat_axis);
        seed = hash_combine(seed, gather_axis);
        seed = hash_combine(seed, indirect);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const kv_cache>(rhs);

        return variable_info == rhs_casted.variable_info &&
               concat_axis == rhs_casted.concat_axis &&
               gather_axis == rhs_casted.gather_axis &&
               indirect == rhs_casted.indirect;
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
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<kv_cache>::load(ib);
        ov::PartialShape data_shape;
        ov::element::Type_t data_type;
        std::string variable_id;
        ib >> variable_id;
        ib >> data_shape;
        ib >> make_data(&data_type, sizeof(ov::element::Type_t));
        variable_info = { data_shape, data_type, variable_id };
        ib >> concat_axis;
        ib >> gather_axis;
        ib >> indirect;
    }
};
}  // namespace cldnn
