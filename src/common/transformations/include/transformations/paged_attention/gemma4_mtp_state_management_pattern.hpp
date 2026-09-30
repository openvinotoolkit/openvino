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
