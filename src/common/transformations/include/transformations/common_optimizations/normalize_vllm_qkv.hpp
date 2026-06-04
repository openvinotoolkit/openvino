// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

class TRANSFORMATIONS_API NormalizeVLLMQKV : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NormalizeVLLMQKV");
    NormalizeVLLMQKV();
};

}  // namespace ov::pass
