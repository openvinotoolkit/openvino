// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// The GPU sizes the internal buffers of a PagedAttention op from its
// max_context_len value. With a static token count the PA output is static, so
// the GPU allocates it and those buffers while it builds the network, before
// any input values exist, and fails.
//
// Gives the queries, keys and values of every PA op a dynamic token count where
// they are model inputs, as in the PA blocks NPUW cuts out of a variant, and
// reshapes the results back, so the ports keep their static shapes. Returns
// false for a model without such a PA op.
class DynamicPATokens : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::DynamicPATokens");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
