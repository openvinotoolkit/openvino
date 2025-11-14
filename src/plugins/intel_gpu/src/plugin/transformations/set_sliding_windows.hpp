// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class SetSlidingWindows: public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SetSlidingWindows");
    SetSlidingWindows() {}
    SetSlidingWindows(const std::string sliding_windows) {
        std::stringstream ss(sliding_windows);
        std::string token;
        while (std::getline(ss, token, ',')) {
            sliding_window_per_layer.push_back(std::stoi(token));
        }
    }

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
    std::vector<size_t> sliding_window_per_layer;
};

}  // namespace ov::intel_gpu
