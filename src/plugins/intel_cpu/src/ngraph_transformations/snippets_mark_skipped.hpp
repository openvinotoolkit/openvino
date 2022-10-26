// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

/**
 * @interface SnippetsMarkSkipped
 * @brief Mark operations that should be ignored by snippets on tokenization stage. A typical example is eltwise operations
 * that will be fused into convolutions on plugin side.
 */
class SnippetsMarkSkipped : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SnippetsMarkSkipped", "0");
    SnippetsMarkSkipped() : ModelPass() {}
    bool run_on_model(const std::shared_ptr<ov::Model> &) override;
};

/*
NotSet - not part of a fusing chain
FusedTerminator - the node is fused, but the chain can't be continued
FusedWithConvolution, FusedWithConvolutionSumActivation, FusedWithMisc - fusing chains with different continuation rules
IgnoredAfterInputs - node must be skipped, since can't be handled properly at this time. Also a continuable fusing chain.
Order of SnippetsNodeType is important!:
* SnippetsNodeType >= FusedTerminator is a Fused chain
* SnippetsNodeType > FusedTerminator is a Fused chain that may be continued
*/
// Todo: Snippets currently support only FP32 precision, however network inputs could be converted to another precision by plugin
//  (in other words after tokenization). To handle this behavior, eltwise chains that start at input are marked as IgnoredAfterInputs.
//  Tis is not a real plugin-side fusing, but rather a workaround to guarantee that snippets always executed in FP32.
enum class NodeFusingType : int64_t {
    NotSet,
    FusedTerminator,
    FusedWithConvolution,  FusedWithBinaryConvolution, FusedWithConvolutionSumActivation,
    FusedWithMatMul, FusedWithMatMulI8, FusedWithReduce, FusedWithMisc, IgnoredAfterInputs};

}   // namespace intel_cpu
}   // namespace ov
