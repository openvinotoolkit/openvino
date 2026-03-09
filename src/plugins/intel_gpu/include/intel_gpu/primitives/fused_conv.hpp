// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include "openvino/op/fused_conv.hpp"
#include "openvino/op/util/variable.hpp"
#include <vector>

namespace cldnn {

using FusedConv = ov::op::FusedConv;

/// @brief fused_conv primitive
/// @details Fuses Gather(beam_idx) + Concat + GroupConv + SiLU + Slice for depthwise causal conv
struct fused_conv : public primitive_base<fused_conv> {
    CLDNN_DECLARE_PRIMITIVE(fused_conv)

    fused_conv() : primitive_base("", {}) {}

    /// @brief Constructs fused_conv primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    fused_conv(const primitive_id& id,
               const std::vector<input_info>& inputs,
               const ov::op::util::VariableInfo& variable_info)
        : primitive_base(id, inputs),
          variable_info(variable_info) {
    }

    ov::op::util::VariableInfo variable_info;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, std::hash<std::string>()(variable_info.variable_id));
        seed = hash_combine(seed, variable_info.data_type.hash());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const fused_conv>(rhs);
        return variable_info == rhs_casted.variable_info;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<fused_conv>::save(ob);
        ov::element::Type_t data_type = variable_info.data_type;
        ob << variable_info.variable_id;
        ob << variable_info.data_shape;
        ob << make_data(&data_type, sizeof(ov::element::Type_t));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<fused_conv>::load(ib);
        ov::PartialShape data_shape;
        ov::element::Type_t data_type = ov::element::Type_t::dynamic;
        std::string variable_id;
        ib >> variable_id;
        ib >> data_shape;
        ib >> make_data(&data_type, sizeof(ov::element::Type_t));
        variable_info = {data_shape, data_type, variable_id};
    }
};

}  // namespace cldnn
