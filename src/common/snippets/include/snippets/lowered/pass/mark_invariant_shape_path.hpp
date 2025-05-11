// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface MarkInvariantShapePath
 * @brief The helper pass for BufferAllocation pipeline:
 *          - Many buffer-relates passes (SetBufferRegGroup, DefineBufferClusters) depend on loop pointer increments.
 *            The increments are unknown in dynamic case, so these passes can't set reg groups and clusters efficiently.
 *            The current pass marks expressions port which will have the same shape. The shape and layout means
 *            the same loop pointer arithmetic in runtime.
 * @ingroup snippets
 */
class MarkInvariantShapePath: public RangedPass {
public:
    OPENVINO_RTTI("MarkInvariantShapePath", "", RangedPass);
    MarkInvariantShapePath() = default;

    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

    /**
     * @brief Returns ID (color) of the current Invariant Shape path for the passed port.
     *        Ports which have the same IDs of the paths - will have the same shapes in runtime.
     *        Note: if passed port is input port, the method returns value for source of port connector
     *              for the passed port. Because the shape is created by output ports of expressions.
     * @param port target expression port
     * @return ID
     */
    static size_t getInvariantPortShapePath(const ExpressionPort& port);

private:
    /**
     * @brief Sets ID (color) of the current Invariant Shape path for the passed output port.
     *        Ports which have the same IDs of the paths - will have the same shapes in runtime.
     * @param port target expression port
     * @param value ID of the path (color)
     */
    static void SetInvariantPortShapePath(const ExpressionPort& port, size_t value);

    /**
     * @brief Return runtime info for the passed expression port
     * @param port target expression port
     * @return runtime info map
     */
    static ov::RTMap& get_rt_info(const ExpressionPort& port);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
