// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

class LoraStatefulToStatelessPass : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::LoraStatefulToStatelessPass");
    explicit LoraStatefulToStatelessPass() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
