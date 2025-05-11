// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SimplifyShapeOfSubGraph;
class TRANSFORMATIONS_API GroupedGatherElimination;
class TRANSFORMATIONS_API GatherNopElimination;
class TRANSFORMATIONS_API SimplifyGatherShapeOf;
class TRANSFORMATIONS_API SimplifySecondInputOfReshape;
class TRANSFORMATIONS_API AbsSinking;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief GroupedGatherElimination transformation replaces group of Gather
 * operations with the first Gather in this group and updated indices input
 * in case all Gathers in the group are consumed by the same Concat in incremental order.
 */
class ov::pass::GroupedGatherElimination : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GroupedGatherElimination");
    GroupedGatherElimination();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief SimplifyShapeOfSubGraph transformation runs specific optimizations of shape sub-graphs
 */
class ov::pass::SimplifyShapeOfSubGraph : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SimplifyShapeOfSubGraph");
    explicit SimplifyShapeOfSubGraph(bool use_shapes = true) : m_use_shapes(use_shapes){};
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool m_use_shapes;
};

/**
 * @ingroup ov_transformation_common_api
 * @brief GatherNopElimination transformation optimizes out useless Gather operations
 */
class ov::pass::GatherNopElimination : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GatherNopElimination");
    GatherNopElimination();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief SimplifyGatherShapeOf optimizes `gather->shapeof` into `shapeof->gather` for 0D indices.
 * Other cases into Concat of shapeof/gather(data) + shapeof(indices) transformation optimizes out
 * useless Gather operations
 */
class ov::pass::SimplifyGatherShapeOf : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SimplifyGatherShapeOf");
    SimplifyGatherShapeOf();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief SimplifySecondInputOfReshape optimizes `shapeof->gather` into zero values for
 * reshape pattern values if possible.
 */
class ov::pass::SimplifySecondInputOfReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SimplifySecondInputOfReshape");
    SimplifySecondInputOfReshape();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief AbsSinking optimizes out the Abs which input is non negative. Has a special case for Concat -> Abs graph, it
 * moves Abs up through Concat to its inputs, tries to constant fold new Abs ops. In case folding fails applies
 * optimization to the leftover Abs ops
 */
class ov::pass::AbsSinking : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("AbsSinking");
    AbsSinking();
};
