// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <unordered_set>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API StateManagementPattern;

}  // namespace pass
}  // namespace ov

class ov::pass::StateManagementPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("StateManagementPattern");

    /// Maps "variable_id/k" or "variable_id/v" to the owning key_cache/value_cache Parameter.
    /// When multiple SDPA layers share the same ReadValue variable_id, the first one
    /// creates the Parameter (owner, write_kv_cache=true) and subsequent ones reuse it
    /// (shared, write_kv_cache=false).
    using KvCacheParamMap = std::map<std::string, std::shared_ptr<ov::op::v0::Parameter>>;

    StateManagementPattern(ov::pass::paged_attention::PaParams& pa_params,
                           ov::pass::paged_attention::PaResults& results,
                           const ov::pass::paged_attention::Options& options,
                           KvCacheParamMap& seen_kv_var_ids);

private:
    int m_layer_index = 0;
};
