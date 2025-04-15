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
 * Move PackPadded through RNN ops, because RNN(PackPadded(x)) == PackPadded(RNN(x)).
 */
class MovePackThroughLstm : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::MovePackThroughLstm");
    MovePackThroughLstm();
};

/**
 * Remove PackPadded -> PadPacked ops.
 */
class RemovePackingOps : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::RemovePackingOps");
    RemovePackingOps();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
