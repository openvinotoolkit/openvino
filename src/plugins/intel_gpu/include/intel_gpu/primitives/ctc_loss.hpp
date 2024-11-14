// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @brief CTCLoss-4 primitive.
struct ctc_loss : primitive_base<ctc_loss> {
    CLDNN_DECLARE_PRIMITIVE(ctc_loss)

    ctc_loss() : primitive_base("", {}) {}

    /// @brief Constructs ctc_loss primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param preprocess_collapse_repeated Flag for preprocessing labels before loss calculation.
    /// @param ctc_merge_repeated Flag for merging repeated characters in a potential alignment.
    /// @param unique Flag to find unique elements in a target.
    ctc_loss(const primitive_id& id,
             const std::vector<input_info>& inputs,
             bool preprocess_collapse_repeated,
             bool ctc_merge_repeated,
             bool unique)
        : primitive_base(id, inputs),
          preprocess_collapse_repeated(preprocess_collapse_repeated),
          ctc_merge_repeated(ctc_merge_repeated),
          unique(unique) {}

    bool preprocess_collapse_repeated = false;
    bool ctc_merge_repeated = false;
    bool unique = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, preprocess_collapse_repeated);
        seed = hash_combine(seed, ctc_merge_repeated);
        seed = hash_combine(seed, unique);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const ctc_loss>(rhs);

        return preprocess_collapse_repeated == rhs_casted.preprocess_collapse_repeated &&
               ctc_merge_repeated == rhs_casted.ctc_merge_repeated &&
               unique == rhs_casted.unique;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<ctc_loss>::save(ob);
        ob << preprocess_collapse_repeated;
        ob << ctc_merge_repeated;
        ob << unique;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<ctc_loss>::load(ib);
        ib >> preprocess_collapse_repeated;
        ib >> ctc_merge_repeated;
        ib >> unique;
    }
};

}  // namespace cldnn
