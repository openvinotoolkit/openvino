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
// drops them, so run this first. For an FQ whose range exceeds the f16 maximum, scale the
// producer's input by 2^-k. Only done when every path from there is linear in the scaled value
// and ends in an L2 normalization, which cancels the factor once its eps is scaled by 2^-2k;
// otherwise the FQ is left as is. Rescaled FQs are marked, so running the pass again is a no-op.
class RescaleFP16OverflowQDQ : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::RescaleFP16OverflowQDQ");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
