// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace intel_cpu {
class ConvertPrecisionI64ToI32: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ConvertPrecisionI64ToI32", "0");

    ConvertPrecisionI64ToI32() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace intel_cpu
}  // namespace ov
