// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov::pass::paged_attention {

// rt_info keys with the KV cache head counts and head sizes, set on PagedAttention nodes.
constexpr const char* NUM_K_HEADS = "num_k_heads";
constexpr const char* K_HEAD_SIZE = "k_head_size";
constexpr const char* NUM_V_HEADS = "num_v_heads";
constexpr const char* V_HEAD_SIZE = "v_head_size";

}  // namespace ov::pass::paged_attention
