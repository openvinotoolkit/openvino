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
 *          - Many buffer-relates passes (SetBufferRegGroup, DefineBufferClusters) depend on loop pointer arithmethic.
 *            In dynamic case they're unknown so these passes may non effieciently set reg groups and clusters.
 *            The current pass marks expressions port which will have the same shape. The shape and layout means
 *            the same loop pointer arithmethic in runtime.
 * @ingroup snippets
 */
class MarkInvariantShapePath: public RangedPass {
public:
    OPENVINO_RTTI("MarkInvariantShapePath", "RangedPass")
    MarkInvariantShapePath() = default;

    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

    static size_t getInvariantPortShapePath(const ExpressionPort& port) {
        auto& rt = get_rt_info(port);
        const auto rinfo = rt.find("InvariantShapePath");
        OPENVINO_ASSERT(rinfo != rt.end(), "Invariant path for this path has not been marked!");
        return rinfo->second.as<size_t>();
    }

private:
    static void SetInvariantPortShapePath(const ExpressionPort& port, size_t value) {
        auto& rt = get_rt_info(port);
        rt["InvariantShapePath"] = value;
    }

    static ov::RTMap& get_rt_info(const ExpressionPort& port) {
        const auto& node = port.get_expr()->get_node();
        const auto port_idx = port.get_index();
        const auto is_input = port.get_type() == ExpressionPort::Input;
        OPENVINO_ASSERT((is_input && (port_idx < node->get_input_size())) || (!is_input && (port_idx < node->get_output_size())),
                        "Node has incompatible port count with the expression");
        return is_input ? node->input(port_idx).get_rt_info() : node->output(port_idx).get_rt_info();
    }
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
