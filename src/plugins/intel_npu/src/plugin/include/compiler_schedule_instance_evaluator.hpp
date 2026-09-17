// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/isection_instance_evaluator.hpp"
#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

/**
 * @brief Evaluator used to "teach" the CRE how to evaluate section instances corresponding to compiler schedules
 * (ELF_MAIN_SCHEDULE_X & DYNAMIC_SCHEDULE_Y).
 * @details The evaluation is performed by sending a compatibility string to the compiler and querying the evaluation
 * result.
 * @note The class follows a singleton pattern since the state of the class and its behavior should not vary at all
 * within the scope of the process.
 */
class CompilerScheduleInstanceEvaluator : public ISectionInstanceEvaluator {
public:
    CompilerScheduleInstanceEvaluator() = delete;
    CompilerScheduleInstanceEvaluator(const CompilerScheduleInstanceEvaluator& other) = delete;
    CompilerScheduleInstanceEvaluator(CompilerScheduleInstanceEvaluator&& other) = delete;
    void operator=(const CompilerScheduleInstanceEvaluator&) = delete;
    void operator=(CompilerScheduleInstanceEvaluator&&) = delete;

    static std::shared_ptr<CompilerScheduleInstanceEvaluator> get_instance(
        const ov::SoPtr<intel_npu::IEngineBackend>& backend,
        const std::shared_ptr<CompilerOptionSupportHelper>& option_support_helper);

    /**
     * @brief Evaluate by sending a compatibility string to the compiler and querying the evaluation
     * result.
     * @return "NOT_APPLICABLE" if the string is empty (because older software versions do not have this feature
     * implemented) or the method failed to query the compiler. "SUPPORTED" or "UNSUPPORTED" if the compiler was queried
     * successfully, according to its reply.
     */
    ov::CompatibilityCheck evaluate(std::string_view runtime_requirements) const override;

protected:
    CompilerScheduleInstanceEvaluator(const ov::SoPtr<intel_npu::IEngineBackend>& backend,
                                      const std::shared_ptr<CompilerOptionSupportHelper>& option_support_helper);

private:
    ov::SoPtr<intel_npu::IEngineBackend> m_backend;
    std::shared_ptr<CompilerOptionSupportHelper> m_option_support_helper;
};

}  // namespace intel_npu
