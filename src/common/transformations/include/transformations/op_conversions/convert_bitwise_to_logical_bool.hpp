// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API ConvertBitwiseAndToLogicalAnd;
class TRANSFORMATIONS_API ConvertBitwiseNotToLogicalNot;
class TRANSFORMATIONS_API ConvertBitwiseOrToLogicalOr;
class TRANSFORMATIONS_API ConvertBitwiseXorToLogicalXor;
}  // namespace pass
}  // namespace ov

class ov::pass::ConvertBitwiseAndToLogicalAnd : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertBitwiseAndToLogicalAnd");
    ConvertBitwiseAndToLogicalAnd();
};
class ov::pass::ConvertBitwiseNotToLogicalNot : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertBitwiseNotToLogicalNot");
    ConvertBitwiseNotToLogicalNot();
};
class ov::pass::ConvertBitwiseOrToLogicalOr : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertBitwiseOrToLogicalOr");
    ConvertBitwiseOrToLogicalOr();
};
class ov::pass::ConvertBitwiseXorToLogicalXor : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertBitwiseXorToLogicalXor");
    ConvertBitwiseXorToLogicalXor();
};
/**
 * @ingroup ov_transformation_common_api
 * @brief Converts Bitwise operators to Logical for boolean datatype for plugins that don't support opset13 Bitwise and
 * to allow for constant folding for bool.
 */
class ConvertBitwiseToLogical : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertBitwiseToLogical");
    ConvertBitwiseToLogical() {
        add_matcher<ov::pass::ConvertBitwiseAndToLogicalAnd>();
        add_matcher<ov::pass::ConvertBitwiseNotToLogicalNot>();
        add_matcher<ov::pass::ConvertBitwiseOrToLogicalOr>();
        add_matcher<ov::pass::ConvertBitwiseXorToLogicalXor>();
    }
};
