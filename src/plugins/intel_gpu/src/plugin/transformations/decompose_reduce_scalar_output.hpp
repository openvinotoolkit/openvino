// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

// In some case, Reduce OP is used to reduce one 2D/3D/4D/5D tensor to a scalar output, which leads to all computation
// are executed in single EU thread due to only one output, then fall in very poor performance. This pattern is used to
// detect this case and decompose Reduce by dimension to avoid poor performance.
class DecomposeReduceForScalarOutput : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeReduceForScalarOutput", "0");
    DecomposeReduceForScalarOutput();
};

}  // namespace intel_gpu
}  // namespace ov
