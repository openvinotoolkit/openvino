// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinking;
class TRANSFORMATIONS_API TransposeConvert;
class TRANSFORMATIONS_API TransposeEltwise;
class TRANSFORMATIONS_API TransposeReduction;
class TRANSFORMATIONS_API TransposeFQReduction;
class TRANSFORMATIONS_API TransposeFuse;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TransposeReduction transformation sinks Transpose through Reduce operations
 */
class ov::pass::TransposeReduction : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeReduction");
    TransposeReduction();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TransposeFQReduction transformation sinks Transpose through FakeQuantize in case it is followed by reduction
 * or squeeze
 */
class ov::pass::TransposeFQReduction : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeFQReduction");
    TransposeFQReduction();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TransposeConvert transformation sinks Transpose through Convert
 */
class ov::pass::TransposeConvert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeConvert");
    TransposeConvert();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TransposeEltwise transformation sinks Transpose through Eltwise
 */
class ov::pass::TransposeEltwise : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeEltwise");
    TransposeEltwise();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TransposeFuse transformation eliminates 2 consequtive Transposes if they result in no changes to input or
 * fuses them to single Transpose if input gets changed
 */
class ov::pass::TransposeFuse : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeFuse");
    TransposeFuse();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TransposeSinking transformation sinks Transposes through known operations
 */
class ov::pass::TransposeSinking : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("TransposeSinking");
    TransposeSinking() {
        add_matcher<ov::pass::TransposeFQReduction>();
        add_matcher<ov::pass::TransposeReduction>();
        add_matcher<ov::pass::TransposeConvert>();
        add_matcher<ov::pass::TransposeEltwise>();
        add_matcher<ov::pass::TransposeFuse>();
    }
};
