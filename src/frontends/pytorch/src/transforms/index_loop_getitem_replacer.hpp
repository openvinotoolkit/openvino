// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::frontend::pytorch::pass {

/**
 * @brief IndexLoopGetitemReplacer transformation replaces following graph:
 * aten::chunk->prim::Loop(aten::__getitem__) to Slice inside the Loop
 */
class IndexLoopGetitemReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::IndexLoopGetitemReplacer");
    IndexLoopGetitemReplacer();
};

}  // namespace ov::frontend::pytorch::pass
