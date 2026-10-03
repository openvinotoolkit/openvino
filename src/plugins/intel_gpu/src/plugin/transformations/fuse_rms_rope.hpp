// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/manager.hpp"

namespace ov::intel_gpu {

inline constexpr char fuse_rms_rope_epsilon_key[] = "intel_gpu_fuse_rms_rope_epsilon";

class FuseRMSRoPE : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("FuseRMSRoPE");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
