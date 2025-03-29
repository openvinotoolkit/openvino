// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct multinomial : public primitive_base<multinomial> {
    CLDNN_DECLARE_PRIMITIVE(multinomial)

    multinomial() : primitive_base("", {}) {}

    /// @brief Constructs multinomial primitive.
    /// @param id This primitive id.
    /// @param cdf Cumulative distribution of probabilties.
    //  @param random_probabilities Random probability samples.
    multinomial(const primitive_id& id,
          const input_info& cdf,
          const input_info& random_probabilities,
          data_types output_data_type,
          bool with_replacement,
          bool log_probs,
          std::uint64_t global_seed,
          std::uint64_t op_seed,
          std::int64_t num_samples)
        : primitive_base{id, {cdf, random_probabilities}},
          output_data_type {output_data_type},
          with_replacement {with_replacement},
          log_probs {log_probs},
          global_seed {global_seed},
          op_seed {global_seed},
          num_samples {num_samples}
    {}

    data_types output_data_type;
    bool with_replacement;
    bool log_probs;
    std::uint64_t global_seed;
    std::uint64_t op_seed;
    std::int64_t num_samples;

    std::size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, output_data_type);
        seed = hash_combine(seed, with_replacement);
        seed = hash_combine(seed, log_probs);
        seed = hash_combine(seed, global_seed);
        seed = hash_combine(seed, op_seed);
        return hash_combine(seed, num_samples);
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        const multinomial& rhs_casted = downcast<const multinomial>(rhs);
        return output_data_type == rhs_casted.output_data_type &&
            with_replacement == rhs_casted.with_replacement &&
            log_probs == rhs_casted.log_probs &&
            global_seed == rhs_casted.global_seed &&
            op_seed == rhs_casted.op_seed &&
            num_samples == rhs_casted.num_samples;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<multinomial>::save(ob);
        ob << ov::element::Type(output_data_type).to_string();
        ob << with_replacement;
        ob << log_probs;
        ob << global_seed;
        ob << op_seed;
        ob << num_samples;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<multinomial>::load(ib);
        std::string data_type;
        ib >> data_type;
        output_data_type = ov::element::Type(data_type);
        ib >> with_replacement;
        ib >> log_probs;
        ib >> global_seed;
        ib >> op_seed;
        ib >> num_samples;
    }
};
}  // namespace cldnn
