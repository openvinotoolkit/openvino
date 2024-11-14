// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace intel_gpu {

class PrintModelStatistics : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("PrintModelStatistics", "0");
    PrintModelStatistics() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace intel_gpu
}  // namespace ov
