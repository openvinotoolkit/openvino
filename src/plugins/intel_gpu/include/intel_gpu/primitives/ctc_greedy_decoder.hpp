// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief CTC greedy decoder primitve
struct ctc_greedy_decoder : public primitive_base<ctc_greedy_decoder> {
    CLDNN_DECLARE_PRIMITIVE(ctc_greedy_decoder)

    /// @brief Constructs ctc_greedy_decoder primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id (input, sequence_indicators, second_output(optional)).
    /// @param blank_index Specifies the class index to use for the blank class.
    /// @param ctc_merge_repeated Flag for merging repeated labels during the CTC calculation
    ctc_greedy_decoder(const primitive_id& id,
                       const std::vector<input_info>& inputs,
                       const uint32_t blank_index,
                       const bool ctc_merge_repeated,
                       const tensor output_tensor,
                       const padding& output_padding = padding())
        : primitive_base(id, inputs, {output_padding})
        , blank_index(blank_index)
        , ctc_merge_repeated(ctc_merge_repeated)
        , output_tensor(output_tensor) {}

    uint32_t blank_index;
    bool ctc_merge_repeated;
    tensor output_tensor;
    primitive_id second_output;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, blank_index);
        seed = hash_combine(seed, ctc_merge_repeated);
        seed = hash_combine(seed, second_output.empty());
        return seed;
    }
};
}  // namespace cldnn
