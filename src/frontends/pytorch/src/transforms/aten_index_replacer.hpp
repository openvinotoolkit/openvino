// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::frontend::pytorch::pass {

// This transformation replaces pattern prim::ListConstruct->aten::index
class AtenIndexToSelect : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::AtenIndexToSelect");
    AtenIndexToSelect();
};

}  // namespace ov::frontend::pytorch::pass
