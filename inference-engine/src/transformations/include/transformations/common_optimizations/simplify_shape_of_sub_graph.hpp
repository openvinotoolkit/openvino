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

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SimplifyShapeOfSubGraph;
class TRANSFORMATIONS_API SharedShapeOf;
class TRANSFORMATIONS_API GroupedGatherElimination;
class TRANSFORMATIONS_API GatherNopElimination;
class TRANSFORMATIONS_API SimplifyGatherShapeOf;

}  // namespace pass
}  // namespace ov


/**
 * @ingroup ie_transformation_common_api
 * @brief SharedShapeOf transformation replaces group of ShapeOf
 * operations with the first ShapeOf in this group. All ShapeOfs in this group
 * must be equal and consume the same output port.
 */
class ov::pass::SharedShapeOf: public ov::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ov::Function> f) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GroupedGatherElimination transformation replaces group of Gather
 * operations with the first Gather in this group and updated indices input
 * in case all Gathers in the group are consumed by the same Concat in incremental order.
 */
class ov::pass::GroupedGatherElimination: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GroupedGatherElimination();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SimplifyShapeOfSubGraph transformation runs specific optimizations of shape sub-graphs
 */
class ov::pass::SimplifyShapeOfSubGraph: public ov::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ov::Function> f) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GatherNopElimination transformation optimizes out useless Gather operations
 */
class ov::pass::GatherNopElimination: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GatherNopElimination();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SimplifyGatherShapeOf optimizes `gather->shapeof` into `shapeof->gather` for 0D indices.
 * Other cases into Concat of shapeof/gather(data) + shapeof(indices) transformation optimizes out
 * useless Gather operations
 */
class ov::pass::SimplifyGatherShapeOf: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SimplifyGatherShapeOf();
};
