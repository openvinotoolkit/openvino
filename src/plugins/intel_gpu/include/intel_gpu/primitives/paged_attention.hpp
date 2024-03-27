// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/optionals.hpp"

#include <vector>

namespace cldnn {

struct paged_attention : public primitive_base<paged_attention> {
    CLDNN_DECLARE_PRIMITIVE(paged_attention)

    paged_attention() : primitive_base("", {}) {}

    paged_attention(const primitive_id& id,
                    const std::vector<input_info>& inputs,
                    const padding& output_padding = padding())
        : primitive_base(id, inputs, {output_padding}) {
        OPENVINO_ASSERT(inputs.size() == 13, "[GPU] Unexpected inputs number for PagedAttention primitive: ", inputs.size());
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<paged_attention>::save(ob);
        ob << head_size;
        ob << heads_num;
        ob << kv_heads_num;
        ob << block_size;
        ob << x_block_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<paged_attention>::load(ib);
        ib >> head_size;
        ib >> heads_num;
        ib >> kv_heads_num;
        ib >> block_size;
        ib >> x_block_size;
    }

    optional_value<float> scale_val{};
    size_t head_size = 0;
    size_t heads_num = 0;
    size_t kv_heads_num = 0;
    size_t block_size = 0;
    size_t x_block_size = 0;
};
}  // namespace cldnn
