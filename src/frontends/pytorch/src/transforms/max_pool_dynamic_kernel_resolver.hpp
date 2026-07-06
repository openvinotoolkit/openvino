// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

// Resolves the deferred max_pool placeholder (a PtFrameworkNode emitted by translate_max_pool_base
// when the kernel_size was not a compile-time constant). It runs after shape propagation, so a
// kernel derived from a runtime size (e.g. F.max_pool2d(x, [1, x.size(3)])) that became statically
// known -- e.g. via convert_model(input=...) -- lowers to the ordinary static v14::MaxPool; a kernel
// that is still dynamic falls back to the ReduceMax full-extent decomposition.
class MaxPoolDynamicKernelResolver : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::MaxPoolDynamicKernelResolver");
    MaxPoolDynamicKernelResolver();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
