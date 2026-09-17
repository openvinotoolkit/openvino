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
 * @brief Evaluator corresponding to a single section instance during CRE evaluation.
 * @details The class was designed to evaluate the given compatibility string in a lazy manner, and then cache the
 * result.
 */
class SingleSectionInstanceEvaluator {
public:
    SingleSectionInstanceEvaluator(const std::shared_ptr<ISectionInstanceEvaluator>& impl,
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
     * @brief The actual evaluator that returns the result given a compatibility string.
     */
    std::shared_ptr<ISectionInstanceEvaluator> m_impl;
    std::string m_runtime_requirements;

    /**
     * @brief If evaluation is performed, the result will be stored here for future use.
     */
    mutable std::optional<ov::CompatibilityCheck> m_result;
};

}  // namespace intel_npu
