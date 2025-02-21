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
    lora(const primitive_id& id,
         const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {}

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<lora>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<lora>::load(ib);
    }
};

}  // namespace cldnn
