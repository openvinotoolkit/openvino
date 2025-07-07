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

    size_t hash() const override {
        size_t seed = primitive::hash();
        membuf mem_buf;
        {
            std::ostream out_mem(&mem_buf);
            BinaryOutputBuffer ob = BinaryOutputBuffer(out_mem);
            save(ob);
        }
        seed = hash_range(seed, mem_buf.begin(), mem_buf.end());

        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<sparse_fill_empty_rows>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<sparse_fill_empty_rows>::load(ib);
    }
};
}  // namespace cldnn
