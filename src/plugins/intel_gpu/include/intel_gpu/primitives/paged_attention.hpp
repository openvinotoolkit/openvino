// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/program.hpp"

#include <vector>

namespace cldnn {

struct paged_attention : public primitive_base<paged_attention> {
    CLDNN_DECLARE_PRIMITIVE(paged_attention)

    static constexpr size_t block_size = 16;

    paged_attention() : primitive_base("", {}) {}

    paged_attention(const primitive_id& id,
                    const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {
        OPENVINO_ASSERT((inputs.size() == 13) || (inputs.size() == 16),
                        "[GPU] Unexpected inputs number for PagedAttention primitive: ",
                        inputs.size());
    }

    bool has_scores_output() const {
        return num_outputs == 2;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const paged_attention>(rhs);

        return head_size == rhs_casted.head_size &&
               heads_num == rhs_casted.heads_num &&
               kv_heads_num == rhs_casted.kv_heads_num &&
               sliding_window == rhs_casted.sliding_window &&
               has_alibi == rhs_casted.has_alibi &&
               has_rotated_blocks == rhs_casted.has_rotated_blocks &&
               scale_val.value_or(1.0f) == rhs_casted.scale_val.value_or(1.0f);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<paged_attention>::save(ob);
        ob << head_size;
        ob << heads_num;
        ob << kv_heads_num;
        ob << has_alibi;
        ob << has_rotated_blocks;
        ob << sliding_window;

        if (scale_val.has_value()) {
            ob << true;
            ob << scale_val.value();
        } else {
            ob << false;
        }
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<paged_attention>::load(ib);
        ib >> head_size;
        ib >> heads_num;
        ib >> kv_heads_num;
        ib >> has_alibi;
        ib >> has_rotated_blocks;
        ib >> sliding_window;

        bool has_scale;
        ib >> has_scale;
        if (has_scale) {
            float scale = 1.0f;
            ib >> scale;
            scale_val = scale;
        } else {
            scale_val = std::optional<float>();
        }
    }

    std::optional<float> scale_val{};
    size_t head_size = 0;
    size_t heads_num = 0;
    size_t kv_heads_num = 0;
    size_t sliding_window = 0;
    bool has_alibi = false;
    bool has_rotated_blocks = false;
};
}  // namespace cldnn
