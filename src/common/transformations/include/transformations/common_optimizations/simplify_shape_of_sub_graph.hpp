// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SimplifyShapeOfSubGraph;
class TRANSFORMATIONS_API GroupedGatherElimination;
class TRANSFORMATIONS_API GatherNopElimination;
class TRANSFORMATIONS_API SimplifyGatherShapeOf;
class TRANSFORMATIONS_API SimplifySecondInputOfReshape;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief GroupedGatherElimination transformation replaces group of Gather
 * operations with the first Gather in this group and updated indices input
 * in case all Gathers in the group are consumed by the same Concat in incremental order.
 */
class ov::pass::GroupedGatherElimination : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GroupedGatherElimination", "0");
    GroupedGatherElimination();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SimplifyShapeOfSubGraph transformation runs specific optimizations of shape sub-graphs
 */
class ov::pass::SimplifyShapeOfSubGraph : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SimplifyShapeOfSubGraph", "0");
    explicit SimplifyShapeOfSubGraph(bool use_shapes = true) : m_use_shapes(use_shapes){};
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool m_use_shapes;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GatherNopElimination transformation optimizes out useless Gather operations
 */
class ov::pass::GatherNopElimination : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherNopElimination", "0");
    GatherNopElimination();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SimplifyGatherShapeOf optimizes `gather->shapeof` into `shapeof->gather` for 0D indices.
 * Other cases into Concat of shapeof/gather(data) + shapeof(indices) transformation optimizes out
 * useless Gather operations
 */
class ov::pass::SimplifyGatherShapeOf : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SimplifyGatherShapeOf", "0");
    SimplifyGatherShapeOf();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SimplifySecondInputOfReshape optimizes `shapeof->gather` into zero values for
 * reshape pattern values if possible.
 */
class ov::pass::SimplifySecondInputOfReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SimplifySecondInputOfReshape", "0");
    SimplifySecondInputOfReshape();
};
