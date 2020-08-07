// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SwishFusion;
class TRANSFORMATIONS_API SwishFusionWithSigmoid;
class TRANSFORMATIONS_API SwishFusionWithSigmoidWithBeta;
class TRANSFORMATIONS_API SwishFusionWithBeta;
class TRANSFORMATIONS_API SwishFusionWithoutBeta;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SwishFusion transformation replaces various sub-graphs to Swish op.
 */
class ngraph::pass::SwishFusion: public ngraph::pass::GraphRewrite {
public:
    SwishFusion() {
        add_matcher<ngraph::pass::SwishFusionWithSigmoid>();
        add_matcher<ngraph::pass::SwishFusionWithSigmoidWithBeta>();
        add_matcher<ngraph::pass::SwishFusionWithBeta>();
        add_matcher<ngraph::pass::SwishFusionWithoutBeta>();
    }
};

class ngraph::pass::SwishFusionWithSigmoid: public ngraph::pass::MatcherPass {
public:
    SwishFusionWithSigmoid();
};

class ngraph::pass::SwishFusionWithSigmoidWithBeta: public ngraph::pass::MatcherPass {
public:
    SwishFusionWithSigmoidWithBeta();
};

class ngraph::pass::SwishFusionWithBeta: public ngraph::pass::MatcherPass {
public:
    SwishFusionWithBeta();
};

class ngraph::pass::SwishFusionWithoutBeta: public ngraph::pass::MatcherPass {
public:
    SwishFusionWithoutBeta();
};
