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
// when kernel_size was not a compile-time constant). Runs after shape propagation: a kernel that
// became static (e.g. via convert_model(input=...)) lowers to a plain v14::MaxPool; one still
// dynamic falls back to the ReduceMax full-extent decomposition.
class MaxPoolDynamicKernelResolver : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::MaxPoolDynamicKernelResolver");
    MaxPoolDynamicKernelResolver();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
