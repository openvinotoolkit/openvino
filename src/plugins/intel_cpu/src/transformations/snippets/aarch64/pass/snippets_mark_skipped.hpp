// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu {

/**
 * @interface SnippetsMarkSkipped
 * @brief Mark operations that should be ignored by snippets on tokenization stage. A typical example is eltwise
 * operations that will be fused into convolutions on plugin side.
 */
class SnippetsMarkSkipped : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SnippetsMarkSkipped");
    SnippetsMarkSkipped() : ModelPass() {}
    bool run_on_model(const std::shared_ptr<ov::Model>&) override;
};

/*
NotSet - not part of a fusing chain
FusedTerminator - the node is fused, but the chain can't be continued
FusedWithConvolution, FusedWithMisc - fusing chains with different continuation rules
IgnoredAfterInputs - node must be skipped, since can't be handled properly at this time. Also a continuable fusing
chain. Order of SnippetsNodeType is important!:
* SnippetsNodeType >= FusedTerminator is a Fused chain
* SnippetsNodeType > FusedTerminator is a Fused chain that may be continued
*/
enum class NodeFusingType : int64_t {
    NotSet,
    FusedTerminator,
    FusedWithConvolution,
    FusedWithBinaryConvolution,
    FusedWithMatMul,
    FusedWithFC,
    FusedWithMisc
};

}  // namespace ov::intel_cpu
