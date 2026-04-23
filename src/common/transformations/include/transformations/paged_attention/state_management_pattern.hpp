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
    StateManagementPattern(ov::pass::paged_attention::PaParams& pa_params,
                           int& layer_index,
                           ov::pass::paged_attention::PaResults& results,
                           const ov::pass::paged_attention::Options& options,
                           std::unordered_set<std::string>& var_ids_to_remove);
};
