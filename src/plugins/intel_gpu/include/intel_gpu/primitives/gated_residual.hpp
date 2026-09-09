// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "intel_gpu/op/gated_residual.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief gated_residual primitive.
/// @details Implements the READ/MIX, WRITE/COMBINE and FINAL_MIX stages of the
/// gated residual (hyper connection) mechanism. See ov::intel_gpu::op::GatedResidual
/// for the exact semantics.
struct gated_residual : public primitive_base<gated_residual> {
    CLDNN_DECLARE_PRIMITIVE(gated_residual)

    using Mode = ov::intel_gpu::op::GatedResidual::Mode;

    gated_residual() : primitive_base("", {}) {}

    gated_residual(const primitive_id& id,
                   const std::vector<input_info>& inputs,
                   Mode mode,
                   size_t hc_count,
                   size_t hidden_size,
                   size_t lowrank,
                   float epsilon,
                   bool norm_gain_plus_one,
                   bool produce_write_gate)
        // READ with produce_write_gate emits (mixed, S); output_data_types and
        // output_paddings must be sized accordingly or get_output_layout(1) is invalid.
        : primitive_base(id,
                         inputs,
                         produce_write_gate ? 2 : 1,
                         std::vector<optional_data_type>(produce_write_gate ? 2 : 1, optional_data_type()),
                         std::vector<padding>(produce_write_gate ? 2 : 1, padding())),
          mode(mode),
          hc_count(hc_count),
          hidden_size(hidden_size),
          lowrank(lowrank),
          epsilon(epsilon),
          norm_gain_plus_one(norm_gain_plus_one),
          produce_write_gate(produce_write_gate) {}

    Mode mode = Mode::READ;
    size_t hc_count = 4;
    size_t hidden_size = 0;
    size_t lowrank = 0;
    float epsilon = 1e-6f;
    bool norm_gain_plus_one = true;
    bool produce_write_gate = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, static_cast<int>(mode));
        seed = hash_combine(seed, hc_count);
        seed = hash_combine(seed, hidden_size);
        seed = hash_combine(seed, lowrank);
        seed = hash_combine(seed, epsilon);
        seed = hash_combine(seed, norm_gain_plus_one);
        seed = hash_combine(seed, produce_write_gate);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const gated_residual>(rhs);
        return mode == rhs_casted.mode && hc_count == rhs_casted.hc_count && hidden_size == rhs_casted.hidden_size &&
               lowrank == rhs_casted.lowrank && epsilon == rhs_casted.epsilon && norm_gain_plus_one == rhs_casted.norm_gain_plus_one &&
               produce_write_gate == rhs_casted.produce_write_gate;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gated_residual>::save(ob);
        ob << make_data(&mode, sizeof(Mode));
        ob << hc_count;
        ob << hidden_size;
        ob << lowrank;
        ob << epsilon;
        ob << norm_gain_plus_one;
        ob << produce_write_gate;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gated_residual>::load(ib);
        ib >> make_data(&mode, sizeof(Mode));
        ib >> hc_count;
        ib >> hidden_size;
        ib >> lowrank;
        ib >> epsilon;
        ib >> norm_gain_plus_one;
        ib >> produce_write_gate;
    }
};

}  // namespace cldnn
