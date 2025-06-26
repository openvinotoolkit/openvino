// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief SparseFillEmptyRows operation.
/// @details Checks the specification for details.
struct SparseFillEmptyRows : public primitive_base<SparseFillEmptyRows> {
    CLDNN_DECLARE_PRIMITIVE(SparseFillEmptyRows)

    SparseFillEmptyRows() : primitive_base("", {}) {}

    /// @brief Constructs SparseFillEmptyRows primitive.
    /// @param id This primitive id.
    /// @param inputs List of input primitives (check specification for details).
    SparseFillEmptyRows(const primitive_id& id,
          const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {}

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
        primitive_base<SparseFillEmptyRows>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<SparseFillEmptyRows>::load(ib);
    }
};
}  // namespace cldnn
