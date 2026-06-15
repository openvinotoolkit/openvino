// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class TransposeFusion: public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("TransposeFusion");
    TransposeFusion(bool supports_immad = false);
};

class TransposeMatMulMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeMatMulMatcher");
    TransposeMatMulMatcher(bool supports_immad);
};

class TransposeMatMulTransposeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeMatMulTransposeMatcher");
    TransposeMatMulTransposeMatcher(bool supports_immad);
};

class TransposeSDPAMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeSDPAMatcher");
    TransposeSDPAMatcher();
};

class TransposeVLSDPAMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeVLSDPAMatcher");
    TransposeVLSDPAMatcher();
};

// QKVSplitReshapeMatcher: Optimize QKV Split in VIT-style attention blocks.
// Matches:   FC -> Reshape([0,0,3,H,S]) -> Transpose([2,0,3,1,4]) -> Split(axis=0,num=3)
//              -> 3x Squeeze(axis=0) -> downstream
// Replaces:  FC -> Reshape([0,0,3,H,S]) -> Split(axis=2,num=3)
//              -> 3x Squeeze(axis=2) -> same downstream
// Effect: crop_axis=2, reshape_axis=2-1=1>=0 -> existing prepare_buffer_fusing path works
class QKVSplitReshapeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QKVSplitReshapeMatcher");
    QKVSplitReshapeMatcher();
};

class TransposeSplitMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeSplitMatcher");
    TransposeSplitMatcher();
};

}   // namespace ov::intel_gpu
