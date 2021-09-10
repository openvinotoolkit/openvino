// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {
class TRANSFORMATIONS_API MatMulHorizontalFusing;
class TRANSFORMATIONS_API SubtractHorizontalFusing;
class TRANSFORMATIONS_API MultiplyHorizontalFusing;
class TRANSFORMATIONS_API AddHorizontalFusing;
class TRANSFORMATIONS_API FakeQuantizeHorizontalFusing;
class TRANSFORMATIONS_API ReshapeHorizontalFusing;
class TRANSFORMATIONS_API OptimizeTransposePairsBeforeMatMul;
class TRANSFORMATIONS_API TransposeHorizontalFusing;
class TRANSFORMATIONS_API HorizontalFusings;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief MatMulHorizontalFusion transformation detects similar matmuls with common parent
 * and tries to fuse it into a single MatMul. This transformation also try to fuse biases.
 */

class ngraph::pass::MatMulHorizontalFusing : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MatMulHorizontalFusing();
};

class ngraph::pass::SubtractHorizontalFusing : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SubtractHorizontalFusing();
};

class ngraph::pass::MultiplyHorizontalFusing : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MultiplyHorizontalFusing();
};

class ngraph::pass::AddHorizontalFusing : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AddHorizontalFusing();
};

class ngraph::pass::FakeQuantizeHorizontalFusing : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    FakeQuantizeHorizontalFusing();
};

class ngraph::pass::ReshapeHorizontalFusing : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReshapeHorizontalFusing();
};

class ngraph::pass::OptimizeTransposePairsBeforeMatMul : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    OptimizeTransposePairsBeforeMatMul();
};

class ngraph::pass::TransposeHorizontalFusing : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeHorizontalFusing();
};

class ngraph::pass::HorizontalFusings : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    HorizontalFusings();
};
