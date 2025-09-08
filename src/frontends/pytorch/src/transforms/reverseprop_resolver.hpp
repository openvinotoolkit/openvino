// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

/**
 * Replace sequence of reverseprop operations with ScatterNdUpdate.
 */
class ReversepropResolver : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::ReversepropResolver");
    ReversepropResolver();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
