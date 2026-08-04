// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <unordered_set>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

class TRANSFORMATIONS_API PagedSelectiveSSMFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedSelectiveSSMFusion");
    PagedSelectiveSSMFusion(ov::pass::paged_attention::PaParams& pa_params,
                            std::unordered_set<std::string>& var_ids_to_remove);

private:
    size_t m_layer_index = 0;
};

}  // namespace ov::pass
