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

    bool isNativelySupported(const ov::Node::type_info_t& type) const;

    std::shared_ptr<ov::Node> changeConstantPrecision(std::shared_ptr<op::v0::Constant>& constant) const;

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace intel_cpu
}  // namespace ov
