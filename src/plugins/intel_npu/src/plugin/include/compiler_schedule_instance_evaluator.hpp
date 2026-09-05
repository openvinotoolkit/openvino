// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/isection_instance_evaluator.hpp"
#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

class CompilerScheduleInstanceEvaluator final : public ISectionInstanceEvaluator {
public:
    CompilerScheduleInstanceEvaluator(const ov::SoPtr<intel_npu::IEngineBackend>& backend,
                                      const std::shared_ptr<CompilerOptionSupportHelper>& option_support_helper);

    ov::CompatibilityCheck evaluate(std::string_view runtime_requirements) const override;

private:
    ov::SoPtr<intel_npu::IEngineBackend> m_backend;
    std::shared_ptr<CompilerOptionSupportHelper> m_option_support_helper;
};

}  // namespace intel_npu
