// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FuseVectorizedMOE2GEMM;
class TRANSFORMATIONS_API FuseVectorizedMOE3GEMM;
class TRANSFORMATIONS_API VectorizedExpertsFusion;

}  // namespace pass
}  // namespace ov

class ov::pass::FuseVectorizedMOE2GEMM : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseVectorizedMOE2GEMM");
    FuseVectorizedMOE2GEMM();
};

class ov::pass::FuseVectorizedMOE3GEMM : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseVectorizedMOE3GEMM");
    FuseVectorizedMOE3GEMM();
};

class ov::pass::VectorizedExpertsFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("VectorizedExpertsFusion");
    VectorizedExpertsFusion() {
        add_matcher<ov::pass::FuseVectorizedMOE2GEMM>();
        add_matcher<ov::pass::FuseVectorizedMOE3GEMM>();
    }
};
