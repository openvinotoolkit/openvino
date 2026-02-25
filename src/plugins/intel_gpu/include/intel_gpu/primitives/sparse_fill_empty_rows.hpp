// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/serialization/utils.hpp"

namespace cldnn {

/// @brief sparse_fill_empty_rows operation.
/// @details Checks the specification for details.
struct sparse_fill_empty_rows : public primitive_base<sparse_fill_empty_rows> {
    CLDNN_DECLARE_PRIMITIVE(sparse_fill_empty_rows)

    sparse_fill_empty_rows() : primitive_base("", {}) {}

    /// @brief Constructs sparse_fill_empty_rows primitive.
    /// @param id This primitive id.
    /// @param inputs List of input primitives (check specification for details).
    sparse_fill_empty_rows(const primitive_id& id,
          const std::vector<input_info>& inputs)
        : primitive_base(id, inputs, 3) {}

    std::vector<float> values;
    std::vector<int64_t> dense_shape;
    std::vector<int64_t> indices;
    float default_value;

    sparse_fill_empty_rows(const primitive_id& id,
                          const std::vector<input_info>& inputs,
                          const std::vector<float>& values,
                          const std::vector<int64_t>& dense_shape,
                          const std::vector<int64_t>& indices,
                          float default_value)
        : primitive_base(id, inputs, 3),
          values(values),
          dense_shape(dense_shape),
          indices(indices),
          default_value(default_value) {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        membuf mem_buf;
        {
            std::ostream out_mem(&mem_buf);
            BinaryOutputBuffer ob = BinaryOutputBuffer(out_mem);
            save(ob);
        }
        seed = hash_range(seed, mem_buf.begin(), mem_buf.end());
        seed = hash_range(seed, indices.begin(), indices.end());
        seed = hash_range(seed, values.begin(), values.end());
        seed = hash_range(seed, dense_shape.begin(), dense_shape.end());
        seed = hash_combine(seed, default_value);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const sparse_fill_empty_rows>(rhs);
        return indices == rhs_casted.indices &&
               values == rhs_casted.values &&
               dense_shape == rhs_casted.dense_shape &&
               default_value == rhs_casted.default_value;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<sparse_fill_empty_rows>::save(ob);
        ob << indices;
        ob << values;
        ob << dense_shape;
        ob << default_value;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<sparse_fill_empty_rows>::load(ib);
        ib >> indices;
        ib >> values;
        ib >> dense_shape;
        ib >> default_value;
    }
};
}  // namespace cldnn
