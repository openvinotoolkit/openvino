// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::frontend::pytorch::pass {

/**
 * Replace sequence of reverseprop operations with ScatterNdUpdate.
 */
class ReversepropResolver : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::frontend::pytorch::pass::ReversepropResolver");
    ReversepropResolver();
};

}  // namespace ov::frontend::pytorch::pass
