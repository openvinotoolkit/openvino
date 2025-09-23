// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/program.hpp"

#include <vector>

namespace cldnn {

#define ENABLE_PA_CM_PATH 1

struct paged_attention : public primitive_base<paged_attention> {
    CLDNN_DECLARE_PRIMITIVE(paged_attention)

    enum PagedAttentionInputIdx {
        QUERY = 0,
        KEY = 1,
        VALUE = 2,
        KEY_CACHE = 3,
        VALUE_CACHE = 4,
        PAST_LENS = 5,
        SUBSEQUENCE_BEGINS = 6,
        BLOCK_INDICES = 7,
        BLOCK_INDICES_BEGINS = 8,
        SCALE = 9,
        SLIDING_WINDOW = 10,
        ALIBI = 11,
        MAX_CONTEXT_LEN = 12,
        SCORE_AGGREGATION = 13,
        ROTATED_BLOCK_INDICES = 14,
        ROTATION_DELTAS = 15,
        ROTATION_TRIG_LUT = 16,
        XATTENTION_THRESHOLD = 17,
        XATTENTION_BLOCK_SIZE = 18,
        XATTENTION_STRIDE = 19,
    };

    static constexpr size_t block_size = 16;
    static constexpr size_t block_size_xattn = 256;

    paged_attention() : primitive_base("", {}) {}

    paged_attention(const primitive_id& id,
                    const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {
        OPENVINO_ASSERT((inputs.size() == 20),
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

        return k_head_size == rhs_casted.k_head_size &&
               v_head_size == rhs_casted.v_head_size &&
               heads_num == rhs_casted.heads_num &&
               kv_heads_num == rhs_casted.kv_heads_num &&
               sliding_window == rhs_casted.sliding_window &&
               has_alibi == rhs_casted.has_alibi &&
               has_score_aggregation == rhs_casted.has_score_aggregation &&
               has_rotated_blocks == rhs_casted.has_rotated_blocks &&
               scale_val.value_or(1.0f) == rhs_casted.scale_val.value_or(1.0f);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<paged_attention>::save(ob);
        ob << k_head_size;
        ob << v_head_size;
        ob << heads_num;
        ob << kv_heads_num;
        ob << has_alibi;
        ob << has_score_aggregation;
        ob << has_rotated_blocks;
        ob << sliding_window;
        ob << has_score_aggregation;
        ob << has_xattention;

        if (scale_val.has_value()) {
            ob << true;
            ob << scale_val.value();
        } else {
            ob << false;
        }
        ob << is_key_by_channel;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<paged_attention>::load(ib);
        ib >> k_head_size;
        ib >> v_head_size;
        ib >> heads_num;
        ib >> kv_heads_num;
        ib >> has_alibi;
        ib >> has_score_aggregation;
        ib >> has_rotated_blocks;
        ib >> sliding_window;
        ib >> has_score_aggregation;
        ib >> has_xattention;

        bool has_scale;
        ib >> has_scale;
        if (has_scale) {
            float scale = 1.0f;
            ib >> scale;
            scale_val = scale;
        } else {
            scale_val = std::optional<float>();
        }
        ib >> is_key_by_channel;
    }

    std::optional<float> scale_val{};
    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t heads_num = 0;
    size_t kv_heads_num = 0;
    size_t sliding_window = 0;
    bool has_alibi = false;
    bool has_rotated_blocks = false;
    bool has_score_aggregation = false;
    bool has_xattention = false;
    bool is_key_by_channel = false;
};
}  // namespace cldnn
