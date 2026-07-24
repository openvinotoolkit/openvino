// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

// decomposes the lietorch SE3 group ops (Exp, Act3) into native opset ops
// they are traced as opaque prim::PythonOp nodes and reach conversion as
// PtFrameworkNodes tagged lietorch::Exp and lietorch::Act3; without this pass
// they fail conversion or fold into an input-invariant constant
class LieTorchSE3Decomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::LieTorchSE3Decomposition");
    LieTorchSE3Decomposition();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
