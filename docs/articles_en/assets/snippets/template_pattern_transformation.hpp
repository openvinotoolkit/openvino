// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace pass {

class DecomposeDivideMatcher;
class ReluReluFusionMatcher;

}  // namespace pass
}  // namespace ov

// ! [graph_rewrite:template_transformation_hpp]
// transformations/template_pattern_transformation.hpp
/**
 * @ingroup ov_transformation_common_api
 * @brief Add transformation description.
 */
class ov::pass::DecomposeDivideMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeDivideMatcher", "0");
    DecomposeDivideMatcher();
};
// ! [graph_rewrite:template_transformation_hpp]

class ov::pass::ReluReluFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReluReluFusionMatcher", "0");
    ReluReluFusionMatcher();
};
