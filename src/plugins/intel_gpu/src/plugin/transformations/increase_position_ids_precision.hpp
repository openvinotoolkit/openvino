// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief This pass adds additional convert nodes on the position_ids input branch (around MatMul operation),
 *        targeting to improve runtime rotary position embeddings calculation for FP16 models.
 *        Lack of these converts leads to lower accuracy in case if long input sequences (larger than 2048)
 *        due to position_ids representation in the FP16 data type.
 */
class IncreasePositionIdsPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IncreasePositionIdsPrecision");
    IncreasePositionIdsPrecision();
};

}   // namespace ov::intel_gpu
