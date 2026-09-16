// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

class TRANSFORMATIONS_API NormalizeVLLMRoPE : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NormalizeVLLMRoPE");
    NormalizeVLLMRoPE();
};

}  // namespace ov::pass
