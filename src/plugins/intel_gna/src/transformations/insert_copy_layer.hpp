// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {
/**
 * @brief Inserts copy layer before ReadValue/Assign (memory) layer if the input is
 * Crop/Concat/Split while skipping Reshape/Trivial transpose/Squeeze/Unsqueeze (Non-functional) layers:
 * [Crop/Concat/Split]         [Crop/Concat/Split]
 *     |                              |
 *     |               =>           [Copy]
 *     |                              |
 * [Memory]                        [Memory]
 *
 * With non-functional layers:

 * [Crop/Concat/Split]         [Crop/Concat/Split]
 *     |                              |
 * [Non-functional]            [Non-functional]
 *     |                              |
 *     |               =>          [Copy]
 *     |                             |
 * [Memory]                       [Memory]
 */
class InsertCopyBeforeAssignLayer : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    InsertCopyBeforeAssignLayer();
};

/**
 * @brief Inserts Copy layer before Concat layer if the input is
 * Crop/Split while skipping Reshape/Trivial transpose/Squeeze/Unsqueeze (non-functional) layers:
 * [Crop/Split]        [Crop/Split]
 *     |                    |
 *     |         =>       [Copy]
 *     |                    |
 * [Concat]             [Concat]
 *
 * With non-functional layers:

 * [Crop/Split]             [Crop/Split]
 *     |                          |
 * [Non-functional]        [Non-functional]
 *     |                          |
 *     |              =>       [Copy]
 *     |                         |
 *  [Concat]                 [Concat]
 * Or if a layer has multiple connections to Concat
 * [any node]         [any node]
 *   |                 |
 *  / \               / \
 *  |  |       =>    |  [Copy]
 *  |  |             |    |
 * [Concat]         [Concat]

 * [any node]         [any node]
 *    |                  |
 *  / \ \             /   \       \
 *  | | |             |  [Copy]  [Copy]
 *  | | |              |    |   /
 * [Concat]            [Concat]
 */
class InsertCopyBeforeConcatLayer : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    InsertCopyBeforeConcatLayer();
};

/**
 * @brief Inserts Copy layer before Broadcast/Tile in two cases:
 * 1. If Parameter is an input to Broadcast/Tile layer.
 *
 *   [Parameter]              [Parameter]
 *     |                          |
 *     |               =>      [Copy]
 *     |                          |
 *   [Broadcast/Tile]       [Broadcast/Tile]
 *
 *
 * 2. If there are Reshape/Trivial transpose/Squeeze/Unsqueeze (non-functional) layers
 *    between Parameter and Broadcast/Tile layer.
 *
 *   [Parameter]              [Parameter]
 *     |                          |
 * [Non functional]    =>   [Non functional]
 *     |                          |
 *     |               =>      [Copy]
 *     |                          |
 *   [Broadcast/Tile]       [Broadcast/Tile]
 *
 * Note: Changes is required due the issue with legacy transformations.
 * Issue is related to renaming of network input layer in case of removing
 * Broadcast/Tile layer. It happens when two following conditions are met:
 * - input layer of Broadcast/Tile is also network input layer (skipping non-functional)
 * - layer is removed from network:
 *       - layer is Broadcast and product of input and target shape is the same
 *       - layer is Tile amd all repeats values are equal 1
 */
class InsertCopyBeforeLayerToBeEliminated : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertCopyBeforeLayerToBeEliminated", "0");
    InsertCopyBeforeLayerToBeEliminated();
};

/**
 * @brief Inserts Copy layer before Concat or ReadValue/Assign (Memory) layer if they use a common input
 * while skipping Reshape/Trivial transpose/Squeeze/Unsqueeze (non-functional) layers
 * ReadValue/Assign layers have priority on Copy insertion:
 *   [Any node]              [Any node]
 *       |                       |
 *     /   \         =>        /   \
 *    |      \              [Copy]  \
 *    |       \               |      \
 * [Memory] [Concat]      [Memory]  [Concat]
 *
 *   [Parameter]              [Parameter]
 *       |                       |
 *     /   \         =>        /   \
 *    |      \              [Copy] [Copy]
 *    |       \               |      \
 * [Memory] [Concat]      [Memory]  [Concat]
 *
 * With non-functional layers:

 *       [Any node]                    [Any node]
 *           |                             |
 *     /            \                 /            \
 * [Non functional]  \     =>   [Non functional]   \
 *    |               \                |           \
 *    |               \             [Copy]         \
 *    |               \               |            \
 * [Memory]       [Concat]      [Memory]        [Concat]
 *
 *       [Parameter]                   [Parameter]
 *           |                             |
 *     /            \               /             \
 * [Non functional]  \     =>   [Non functional]   \
 *    |               \                |           \
 *    |               \             [Copy]       [Copy]
 *    |               \               |            \
 * [Memory]       [Concat]      [Memory]        [Concat]
 */
class HandleMultiConnectedLayerToConcatAndMemory : public ov::pass::ModelPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};

/**
 * @brief Matches the Reshape/Trivial transpose/Squeeze/Unsqueeze (Non-functional), Crop, Split and passes the rt_info
 * to inputs nodes to identify the subgraph which contains only layers mentioned above. If find the parameter with
 * non-functional rt_info, then inserts copy layer in subgraph: [Parameter]         [Parameter] |                    |
 *     |                  [Copy]
 *  [Reshape]    =>         |
 *     |                [Reshape]
 *     |                    |
 * [Result]             [Result]
 *
 *  [Parameter]              [Parameter]
 *       |                        |
 *    [Reshape]               [Reshape]
 *        |                      |
 *     |      \                /     \
 *     |       \       =>  [Copy]     \
 * [Reshape]   [Relu]         |        \
 *     |         |         [Reshape]   [Relu]
 *     |         |            |           \
 * [Result]   [Result]     [Result]     [Result]
 */
class MatchNonComputationalLayers : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MatchNonComputationalLayers();
};

/**
 * @brief Runs MatchNonComputationalLayers transformation in reverse order to passthru rt_info and identify the
 * non-computational subgraphs.
 */
class HandleNonFunctionalSubgraphs : public ngraph::pass::BackwardGraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    HandleNonFunctionalSubgraphs() {
        add_matcher<MatchNonComputationalLayers>();
    }
};

class HandleNonFunctionalSubgraphsCleanup : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("HandleNonFunctionalSubgraphsCleanup", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
