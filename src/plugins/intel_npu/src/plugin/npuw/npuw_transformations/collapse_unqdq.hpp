// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::npuw {

class CollapseUNQDQ : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::CollapseUNQDQ");
    CollapseUNQDQ();
};

// FakeQuantize ranges are the only record of how large an activation gets, and CollapseUNQDQ
// drops them, so run this first. Only the chain UNQDQ -> MatMul -> UNQDQ -> L2 normalization is
// handled, with every intermediate value feeding only the next step. If either FQ range exceeds the
// f16 maximum, the MatMul input (or the input of the first chain's producer, if that chain
// overflows too) is scaled by 2^-k; the normalization cancels the factor once its eps is scaled by
// 2^-2k. Both chains are collapsed together with the rescale, since their calibrated FQ bounds would
// no longer fit the scaled values; the rest of the model is left for CollapseUNQDQ.
class RescaleFP16OverflowQDQ : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::RescaleFP16OverflowQDQ");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
