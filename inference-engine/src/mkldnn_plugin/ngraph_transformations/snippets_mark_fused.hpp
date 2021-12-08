// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace MKLDNNPlugin {
/**
 * @interface SnippetsMarkFused
 * @brief Mark operations that will be fused on plugin side (but not yet in snippets) so they'll be ignored by snippets.
 */
class SnippetsMarkFused : public ov::pass::ModelPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SnippetsMarkFused() : ModelPass() {}
    bool run_on_model(std::shared_ptr<ov::Model> model) override;
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
    FusedWithMatMul, FusedWithMisc, IgnoredAfterInputs};
}  // namespace MKLDNNPlugin