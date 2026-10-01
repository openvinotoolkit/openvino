// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "openvino/op/parameter.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API Gemma4MTPStateManagementPattern;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Converts SDPA of a Gemma4 MTP draft model into PagedAttention that reads the target model's KV cache.
 *
 * The draft has no K/V projections and no state: K/V come as model inputs holding the target's cache.
 * They become key_cache.N / value_cache.N (shared by all layers reading the same input), and the last
 * cached token is read back from them as the current K/V.
 *
 * Before:
 *
 *   +-------+   +-----------------+   +-----------------+   +------+
 *   |   Q   |   | K [B,Hkv,L,S]   |   | V [B,Hkv,L,S]   |   | mask |
 *   +-------+   +-----------------+   +-----------------+   +------+
 *       |                |                     |                |
 *       |       +-----------------+   +-----------------+       |
 *       |       |    repeat_kv    |   |    repeat_kv    |       |
 *       |       +-----------------+   +-----------------+       |
 *       v                v                     v                v
 *   +--------------------------------------------------------------+
 *   |                  ScaledDotProductAttention                   |
 *   +--------------------------------------------------------------+
 *
 * After:
 *
 *   +-------+   +-------------+   +-----------+   +----------------+   +---------------+
 *   |   Q   |   | key_cache.N |   | past_lens |   | block_indices* |   | value_cache.N |
 *   +-------+   +-------------+   +-----------+   +----------------+   +---------------+
 *       |          |      |             |                 |                |        |
 *       |          |      |             v                 v                |        |
 *       |          |      |   +----------------------------------------+   |        |
 *       |          |      +-->|     last token extraction subgraph     |<--+        |
 *       |          |          +----------------------------------------+            |
 *       |          |                              |                                 |
 *       |          |                              | K, V, past_lens - 1             |
 *       v          v                              v                                 v
 *   +----------------------------------------------------------------------------------+
 *   |                                  PagedAttention                                  |
 *   +----------------------------------------------------------------------------------+
 */
class ov::pass::Gemma4MTPStateManagementPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Gemma4MTPStateManagementPattern");

    Gemma4MTPStateManagementPattern(ov::pass::paged_attention::PaParams& pa_params,
                                    std::unordered_set<std::string>& params_to_remove);

private:
    int m_layer_index = 0;
    // K/V model input tensor name -> key_cache.N / value_cache.N created for it.
    std::unordered_map<std::string, std::string> m_input_to_cache;
};
