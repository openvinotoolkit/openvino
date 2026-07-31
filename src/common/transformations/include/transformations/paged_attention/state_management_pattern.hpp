// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <unordered_map>
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

    struct KvCacheParams {
        std::shared_ptr<ov::op::v0::Parameter> k;
        std::shared_ptr<ov::op::v0::Parameter> v;
        bool write_kv_cache;
    };

    // Maps ReadValue variable_id (+ /k or /v suffix) to the parameter name (key_cache.N or value_cache.N).
    using ReadValueToParamMap = std::unordered_map<std::string, std::string>;

    StateManagementPattern(ov::pass::paged_attention::PaParams& pa_params,
                           ov::pass::paged_attention::PaResults& results,
                           const ov::pass::paged_attention::Options& options,
                           std::unordered_set<std::string>& var_ids_to_remove);

private:
    KvCacheParams find_or_create_kv_params(const std::shared_ptr<ov::op::util::ReadValueBase>& k_rv,
                                           const std::shared_ptr<ov::op::util::ReadValueBase>& v_rv,
                                           ov::pass::paged_attention::PaParams& pa_params);

    int m_layer_index = 0;
    ReadValueToParamMap m_read_value_to_params;
};
