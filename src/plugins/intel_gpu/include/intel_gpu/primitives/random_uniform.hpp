// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @brief RandomUniform-8 primitive
/// @details
struct random_uniform : public primitive_base<random_uniform> {
    CLDNN_DECLARE_PRIMITIVE(random_uniform)

    random_uniform() : primitive_base("", {}),
                       global_seed(0),
                       op_seed(0),
                       output_shape{} {}

    /**
     * Construct Random Uniform privitive.
     * @param id primitive id
     * @param inputs inputs parameters ids
     * @param data_type output values data type
     * @param global_seed, op_seed random uniform seed attributes
     * @param output_shape output data shape
     * @param output_format output data shape format
     */
    random_uniform(const primitive_id &id, const std::vector<input_info> &inputs,
                   const data_types &data_type, const uint64_t global_seed,
                   const uint64_t op_seed, const ov::Shape output_shape)
            : primitive_base(id, inputs, 1, {optional_data_type{data_type}}),
              global_seed(global_seed),
              op_seed(op_seed),
              output_shape(output_shape) {}

    random_uniform(const primitive_id &id, const std::vector<input_info> &inputs,
                   const data_types &data_type, const uint64_t global_seed, const uint64_t op_seed)
            : primitive_base(id, inputs, 1, {optional_data_type{data_type}}),
              global_seed(global_seed),
              op_seed(op_seed),
              output_shape() {}

    const uint64_t global_seed;
    const uint64_t op_seed;
    const ov::Shape output_shape;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, global_seed);
        seed = hash_combine(seed, op_seed);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const random_uniform>(rhs);

        return global_seed == rhs_casted.global_seed &&
               op_seed == rhs_casted.op_seed;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<random_uniform>::save(ob);
        ob << global_seed;
        ob << op_seed;
        ob << output_shape;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<random_uniform>::load(ib);
        ib >> *const_cast<uint64_t*>(&global_seed);
        ib >> *const_cast<uint64_t*>(&op_seed);
        ib >> *const_cast<ov::Shape*>(&output_shape);
    }
};

}  // namespace cldnn
