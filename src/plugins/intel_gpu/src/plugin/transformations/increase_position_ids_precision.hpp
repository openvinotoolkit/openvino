// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class IncreasePositionIdsPrecisionForRoPE : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IncreasePositionIdsPrecisionBase");
    IncreasePositionIdsPrecisionForRoPE();
};

class IncreasePositionIdsPrecisionForQwen25VL : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IncreasePositionIdsPrecisionForQwen25VL");
    IncreasePositionIdsPrecisionForQwen25VL();
};

class IncreasePositionIdsPrecisionForLtxVideo : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IncreasePositionIdsPrecisionForLtxVideo");
    IncreasePositionIdsPrecisionForLtxVideo();
};

class IncreasePositionIdsPrecisionForGPTOSS : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IncreasePositionIdsPrecisionForGPTOSS");
    IncreasePositionIdsPrecisionForGPTOSS();
};


/**
 * @brief This pass adds additional convert nodes on the position_ids input branch (around MatMul or Multiply operation),
 *        targeting to improve runtime rotary position embeddings calculation for FP16 models.
 *        Lack of these converts leads to lower accuracy in case if long input sequences (larger than 2048)
 *        due to position_ids representation in the FP16 data type.
 */
class IncreasePositionIdsPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("IncreasePositionIdsPrecision");
    IncreasePositionIdsPrecision();
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

class DisableFP16ComForGPTOSSROPEPattern: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16ComForROPEPattern");
    DisableFP16ComForGPTOSSROPEPattern();
};
}   // namespace ov::intel_gpu
