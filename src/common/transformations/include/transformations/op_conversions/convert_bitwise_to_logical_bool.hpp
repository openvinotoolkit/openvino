// Copyright (C) 2018-2023 Intel Corporation
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
    OPENVINO_RTTI("ConvertBitwiseAndToLogicalAnd", "0");
    ConvertBitwiseAndToLogicalAnd();
};
class ov::pass::ConvertBitwiseNotToLogicalNot : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertBitwiseNotToLogicalNot", "0");
    ConvertBitwiseNotToLogicalNot();
};
class ov::pass::ConvertBitwiseOrToLogicalOr : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertBitwiseOrToLogicalOr", "0");
    ConvertBitwiseOrToLogicalOr();
};
class ov::pass::ConvertBitwiseXorToLogicalXor : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertBitwiseXorToLogicalXor", "0");
    ConvertBitwiseXorToLogicalXor();
};
/**
 * @ingroup ie_transformation_common_api
 * @brief Converts Bitwise operators to Logical for boolean datatype for plugins that don't support opset13 Bitwise
 */
class ConvertBitwiseToLogical : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertBitwiseToLogical", "0");
    ConvertBitwiseToLogical() {
        add_matcher<ov::pass::ConvertBitwiseAndToLogicalAnd>();
        add_matcher<ov::pass::ConvertBitwiseNotToLogicalNot>();
        add_matcher<ov::pass::ConvertBitwiseOrToLogicalOr>();
        add_matcher<ov::pass::ConvertBitwiseXorToLogicalXor>();
    }
};
