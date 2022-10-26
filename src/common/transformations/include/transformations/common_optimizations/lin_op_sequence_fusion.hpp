// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API LinOpSequenceFusion;
class TRANSFORMATIONS_API AddMultiplyFusion;
class TRANSFORMATIONS_API AddAddFusion;
class TRANSFORMATIONS_API MultiplyMultiplyFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::AddMultiplyFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("AddMultiplyFusion", "0");
    AddMultiplyFusion();
};

class ngraph::pass::AddAddFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("AddAddFusion", "0");
    AddAddFusion();
};

class ngraph::pass::MultiplyMultiplyFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MultiplyMultiplyFusion", "0");
    MultiplyMultiplyFusion();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief LinOpSequenceFusion transformation fuses linear operation sequence.
 */
class ngraph::pass::LinOpSequenceFusion : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("LinOpSequenceFusion", "0");
    LinOpSequenceFusion() {
        add_matcher<ngraph::pass::AddMultiplyFusion>();
        add_matcher<ngraph::pass::AddAddFusion>();
        add_matcher<ngraph::pass::MultiplyMultiplyFusion>();
    }
};
