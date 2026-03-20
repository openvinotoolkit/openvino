// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

// Apply DEVICE_ROUTED MoE transformations to models
class ApplyMoEDeviceRoutedTransforms : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ApplyMoEDeviceRoutedTransforms");
    ApplyMoEDeviceRoutedTransforms() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
