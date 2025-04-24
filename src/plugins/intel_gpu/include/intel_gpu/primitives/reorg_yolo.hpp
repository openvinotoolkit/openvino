// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Normalizes results so they sum to 1.
/// @details
/// @par Algorithm:
/// @par Where:
struct reorg_yolo : public primitive_base<reorg_yolo> {
    CLDNN_DECLARE_PRIMITIVE(reorg_yolo)

    reorg_yolo() : primitive_base("", {}) {}

    /// @brief Constructs region_yolo primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param dimension Defines a scope of normalization (see #dimension).
    reorg_yolo(const primitive_id& id,
               const input_info& input,
               const uint32_t stride)
        : primitive_base(id, {input}), stride(stride) {}

    /// @brief Defines a scope of a reorg yolo normalization
    /// @details
    /// Specific behaviour is determined by these parameters, as follows:
    uint32_t stride = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, stride);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const reorg_yolo>(rhs);

        return stride == rhs_casted.stride;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<reorg_yolo>::save(ob);
        ob << stride;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<reorg_yolo>::load(ib);
        ib >> stride;
    }
};
}  // namespace cldnn
#pragma once
