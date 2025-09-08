// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "openvino/op/scatter_elements_update.hpp"

namespace cldnn {

using ScatterElementsUpdateOp = ov::op::v12::ScatterElementsUpdate;

/// @brief
/// @details
struct scatter_elements_update : public primitive_base<scatter_elements_update> {
    CLDNN_DECLARE_PRIMITIVE(scatter_elements_update)

    scatter_elements_update() : primitive_base("", {}) {}

    /// @brief Constructs scatter_elements_update primitive.
    /// @param id This primitive id.
    /// @param dict Input data primitive id.
    /// @param idx Input indexes primitive id.
    /// @param idupd Input updates primitive id.
    /// @param axis Gathering axis.
    /// @param mode Reduction mode.
    scatter_elements_update(const primitive_id& id,
                            const input_info& data,
                            const input_info& idx,
                            const input_info& idupd,
                            const int64_t axis,
                            const ScatterElementsUpdateOp::Reduction mode = ScatterElementsUpdateOp::Reduction::NONE,
                            const bool use_init_val = true)
        : primitive_base(id, {data, idx, idupd}), axis(axis), mode(mode), use_init_val(use_init_val) {}

    /// @brief ScatterElementsUpdate axis
    int64_t axis{0};
    /// @brief Reduction mode
    ScatterElementsUpdateOp::Reduction mode{ScatterElementsUpdateOp::Reduction::NONE};
    /// @brief Use initial value for reduction
    bool use_init_val{true};

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, mode);
        seed = hash_combine(seed, use_init_val);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const scatter_elements_update>(rhs);

        return axis == rhs_casted.axis && mode == rhs_casted.mode
                    && use_init_val == rhs_casted.use_init_val;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<scatter_elements_update>::save(ob);
        ob << axis;
        ob << make_data(&mode, sizeof(ScatterElementsUpdateOp::Reduction));
        ob << use_init_val;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<scatter_elements_update>::load(ib);
        ib >> axis;
        ib >> make_data(&mode, sizeof(ScatterElementsUpdateOp::Reduction));
        ib >> use_init_val;
    }
};
}  // namespace cldnn
