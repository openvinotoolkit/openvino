// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief
 * Moves Gather layer forward from the start to the end of the graph
 * through the unary operations UnaryElementwiseArithmetic, Clamp, Elu, SoftPlus, LogicalNot, Convert
 *
 *  Gather          Unary
 *    |      =>       |
 *  Unary           Gather
 *    |               |
 *  Another         Another
 *
 *   Gather                Unary
 *     |            =>       |
 *   Unary                Gather
 *    |  |                 |   |
 * Any1  Any2             Any1 Any2
 *
 *     Gather                Unary1
 *       |           =>      |   |
 *     Unary1              Unary2 Unary3
 *     |    |                |     |
 * Unary2  Unary3          Gather Gather
 *
 *     Another1              Another1
 *        |                  |      |
 *     Gather             Unary   Gather
 *     |    |         =>     |         |
 *    Unary Another2       Gather     Another2
 *     |                     |
 *    Another3              Another3
 *
 * All GatherSinking tranformations are designed to work in 2 steps:
 * - forward push
 * - backward push
 * Add flag into Gather layer rt_info that prevents backward sinking if the next layer
 * after Gather does not support by GatherSinking transformations. That is done to
 * prevent backward pushing the layer that already pushed forward through the graph.
 */
class GatherSinkingUnaryForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingUnaryForward", "0");
    GatherSinkingUnaryForward();
};

/**
 * @brief
 * Moves Gather layer backward from the end to the start of the graph
 * Works only with single consumer case. If Gather is marked as not-sinkable
 * (since it was moved previously by forward sinking) it is not proceeded.
 *
 *   Any         Any
 *    |           |
 *   Unary  =>   Gather
 *    |           |
 *   Gather      Unary
 */
class GatherSinkingUnaryBackwardSingleConsumer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingUnaryBackwardSingleConsumer", "0");
    GatherSinkingUnaryBackwardSingleConsumer();
};

/**
 * @brief
 * Moves Gather layer backward from the end to the start of the graph
 * Works only with multiple consumer case. If Gather is marked as non-sinkable
 * (since it was moved previously by forward sinking) it is not proceeded.
 *
 *      Any1          Any1
 *       |            |
 *     Unary  =>     Gather
 *     |    |          |
 *   Gather Gather   Unary
 *     |     |       |   |
 *    Any2  Any3    Any2 Any3
 *
 * Moves Gather layer backward only if:
 * - Gather is not marked as non-sinkable
 * - Unary layer has > 1 gather consumers
 * - All Unary consumers are Gather layers
 * - All that Gather layers equal each other
 */
class GatherSinkingUnaryBackwardMultiConsumers : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingUnaryBackwardMultiConsumers", "0");
    GatherSinkingUnaryBackwardMultiConsumers();
};

/**
 * @brief
 * GatherSinkingUnaryBackward transformations calls GatherSinkingUnaryBackward and
 * GatherSinkingUnaryBackwardMultiConsumers so there is no need to use them if GatherSinkingUnaryBackward is used
 */
class GatherSinkingUnaryBackward : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("GatherSinkingUnaryBackward", "0");
    GatherSinkingUnaryBackward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov