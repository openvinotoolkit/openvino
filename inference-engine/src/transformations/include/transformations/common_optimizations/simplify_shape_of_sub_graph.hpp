// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/util.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SimplifyShapeOfSubGraph;
class TRANSFORMATIONS_API SharedShapeOf;
class TRANSFORMATIONS_API GroupedGatherElimination;
class TRANSFORMATIONS_API GatherNopElimination;

}  // namespace pass
}  // namespace ngraph


/**
 * @ingroup ie_transformation_common_api
 * @brief SharedShapeOf transformation replaces group of ShapeOf
 * operations with the first ShapeOf in this group. All ShapeOfs in this group
 * must be equal and consume the same output port.
 */
class ngraph::pass::SharedShapeOf: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GroupedGatherElimination transformation replaces group of Gather
 * operations with the first Gather in this group and updated indices input
 * in case all Gathers in the group are consumed by the same Concat in incremental order.
 */
class ngraph::pass::GroupedGatherElimination: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GroupedGatherElimination();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SimplifyShapeOfSubGraph transformation runs specific optimizations of shape sub-graphs
 */
class ngraph::pass::SimplifyShapeOfSubGraph: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GatherNopElimination transformation optimizes out useless Gather operations
 */
class ngraph::pass::GatherNopElimination: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GatherNopElimination();
};
