// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Inserts copy layer before concat if layer has multiple connection to the same concat:
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
class HandleMultiConnectedLayerToConcat : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HandleMultiConnectedLayerToConcat();
};

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
class InsertCopyBeforeMemoryLayer : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    InsertCopyBeforeMemoryLayer();
};

/**
 * @brief Inserts copy layer before concat layer if the input is
 * Crop/Split while skipping Reshape/Trivial transpose/Squeeze/Unsqueeze (Non-functional) layers:
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
 */
class InsertCopyBeforeConcatLayer : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    InsertCopyBeforeConcatLayer();
};

/**
 * @brief Inserts copy layer before Concat or ReadValue/Assign(memory) layer if the input is the same for them.
 * While skipping Reshape/Trivial transpose/Squeeze/Unsqueeze (Non-functional) layers.
 * Memory layers have priority on copy insertion:
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
class HandleMultiConnectedLayerToConcatAndMemory : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};


/**
 * @brief Matching the Reshape/Trivial transpose/Squeeze/Unsqueeze (Non-functional), Crop, Split and passes the rt_info to inputs nodes
 * to identify the subgraph which contains only layers mentioned above. If find the parameter with non-functional rt_info, than inserts:
 * Inserts copy layer in subgraph.
 * [Parameter]         [Parameter]
 *     |                    |
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

class MatchNonFunctionalLayers : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MatchNonFunctionalLayers();
};

/**
 * @brief Runs MatchNonFunctionalLayers transformation in reverse order to passthru rt_info and identify the noncomputing subgraphs.
 */
class HandleNonComputationalSubgraphs : public ngraph::pass::BackwardGraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    HandleNonComputationalSubgraphs() {
        add_matcher<MatchNonFunctionalLayers>();
    }
};

} // namespace GNAPluginNS

