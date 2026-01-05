// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {

class OPENVINO_API PAToPAWithQQBias : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PAToPAWithQQBias");

    PAToPAWithQQBias() { };
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
}  // namespace pass
}  // namespace ov
