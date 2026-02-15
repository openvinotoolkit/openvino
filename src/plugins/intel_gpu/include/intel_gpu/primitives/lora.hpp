// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

struct lora : public primitive_base<lora> {
    CLDNN_DECLARE_PRIMITIVE(lora);

    lora() : primitive_base("", {}) {}

    /// @brief Constructs LoRA primitive
    /// @param id This primitive id
    /// @param inputs Fixed order inputs:
    /// 1) Main flow input (to which LoRA added)
    /// 2) LoRA input (to which LoRA applied)
    /// 3) Low-rank A matrix
    /// 4) Scale vector alpha
    /// 5) Low-rank B matrix
    /// In case of fused LoRA additional inputs (like 3-5) are added for each fused LoRA accordingly
    /// @param transposed_states Defines whether matrices 3 and 5 from the input are transposed or not
    lora(const primitive_id& id,
         const std::vector<input_info>& inputs,
         bool transposed_states)
        : primitive_base(id, inputs),
          transposed_states(transposed_states) {}

    bool transposed_states;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, transposed_states);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const lora>(rhs);
        return transposed_states == rhs_casted.transposed_states;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<lora>::save(ob);
        ob << transposed_states;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<lora>::load(ib);
        ib >> transposed_states;
    }
};

}  // namespace cldnn
