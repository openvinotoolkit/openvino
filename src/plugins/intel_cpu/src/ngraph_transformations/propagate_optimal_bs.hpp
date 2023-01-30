// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/pass.hpp>

namespace ov {
namespace intel_cpu {
namespace mixed_affinity {
class PropagateOptimalBS : public ov::pass::ModelPass {
public:
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace mixed_affinity
}  // namespace intel_cpu
}  // namespace ov
