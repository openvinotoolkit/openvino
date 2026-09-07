// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string_view>

#include "intel_npu/common/isection.hpp"
#include "intel_npu/common/isection_instance_evaluator.hpp"

namespace intel_npu {

/**
 * @brief Interface that standardizes the evaluation of section types support.
 */
class SectionInstanceEvaluator {
public:
    SectionInstanceEvaluator(const std::shared_ptr<ISectionInstanceEvaluator>& impl,
                             std::string_view runtime_requirements);

    /**
     * @brief Checks whether or not the NPU plugin supports the section instance.
     * @details After evaluation, the result is stored for future use.
     */
    ov::CompatibilityCheck get_result() const;

    /**
     * @brief Tells whether or not the section type instance has been already evaluated.
     */
    bool evaluated() const;

private:
    /**
     * @brief TODO
     */
    std::shared_ptr<ISectionInstanceEvaluator> m_impl;
    std::string m_runtime_requirements;

    /**
     * @brief If evaluation is performed, the result will be stored here for future use.
     */
    mutable std::optional<ov::CompatibilityCheck> m_result;
};

}  // namespace intel_npu
