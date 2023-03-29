// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"
#include "snippets/tensor_descriptor.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface LoopSplitting
 * @brief If a parent loop has larger increment but similar works amount, then split an outer loop into two
 *        so the outermost of the two could be fused with the parent loop.
 * @ingroup snippets
 */
class LoopSplitting : public LinearIRTransformation {
public:
    OPENVINO_RTTI("LoopSplitting", "LinearIRTransformation")
    LoopSplitting();
    bool run(LoweredExprIR& linear_ir) override;

private:
    static bool must_be_split(const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& current,
                                 const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& target);
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
