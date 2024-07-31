// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief CTC greedy decoder primitve
struct ctc_greedy_decoder : public primitive_base<ctc_greedy_decoder> {
    CLDNN_DECLARE_PRIMITIVE(ctc_greedy_decoder)

    ctc_greedy_decoder() : primitive_base("", {}) {}

    /// @brief Constructs ctc_greedy_decoder primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id (input, sequence_indicators, second_output(optional)).
    /// @param blank_index Specifies the class index to use for the blank class.
    /// @param ctc_merge_repeated Flag for merging repeated labels during the CTC calculation
    ctc_greedy_decoder(const primitive_id& id,
                       const std::vector<input_info>& inputs,
                       const uint32_t blank_index,
                       const bool ctc_merge_repeated,
                       const tensor output_tensor)
        : primitive_base(id, inputs)
        , blank_index(blank_index)
        , ctc_merge_repeated(ctc_merge_repeated)
        , output_tensor(output_tensor) {}

    /// @brief Constructs ctc_greedy_decoder primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id (input, sequence_indicators, blank_index(optional)).
    /// @param ctc_merge_repeated Flag for merging repeated labels during the CTC calculation
    ctc_greedy_decoder(const primitive_id& id,
                       const std::vector<input_info>& inputs,
                       const uint32_t blank_index,
                       const bool ctc_merge_repeated,
                       data_types output_data_type = data_types::i32,
                       const size_t num_outputs = 1)
        : primitive_base(id, inputs, num_outputs, {optional_data_type{output_data_type}})
        , blank_index(blank_index)
        , ctc_merge_repeated(ctc_merge_repeated) {}

    uint32_t blank_index = UINT32_MAX;
    bool ctc_merge_repeated = false;
    tensor output_tensor;
    primitive_id second_output;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, blank_index);
        seed = hash_combine(seed, ctc_merge_repeated);
        seed = hash_combine(seed, second_output.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const ctc_greedy_decoder>(rhs);

        return blank_index == rhs_casted.blank_index &&
               ctc_merge_repeated == rhs_casted.ctc_merge_repeated &&
               second_output.empty() == rhs_casted.second_output.empty();
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<ctc_greedy_decoder>::save(ob);
        ob << blank_index;
        ob << ctc_merge_repeated;
        ob << output_tensor;
        ob << second_output;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<ctc_greedy_decoder>::load(ib);
        ib >> blank_index;
        ib >> ctc_merge_repeated;
        ib >> output_tensor;
        ib >> second_output;
    }
};
}  // namespace cldnn
