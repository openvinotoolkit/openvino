// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

namespace intel_npu {

/**
 * @brief Abstract class that standardizes the evaluation of section types support in a lazy manner.
 * @note This type of evaluators are meant to be passed to @ref RuntimeRequirements::check_compatibility. Doing so,
 * will "teach" the CRE how to evaluate the support for that section type at evaluation time.
 */
class ISectionTypeEvaluator {
public:
    virtual ~ISectionTypeEvaluator() = default;

    /**
     * @brief Checks whether or not the NPU plugin supports the section type.
     * @details The evaluation depends on the implementation of the "evaluate" method. After evaluation, the
     * result is stored in @ref m_supported for future use.
     */
    bool get_result() const;

    /**
     * @brief Tells whether or not the section type has already been evaluated.
     */
    bool evaluated() const;

protected:
    ISectionTypeEvaluator() = default;

private:
    /**
     * @brief Checks whether or not the NPU plugin supports the section type. This function will be called at most once
     * during execution, in which case the result will be stored for future use.
     * @note This method will typically be called only if the corresponding CRE token has to be evaluated during CRE
     * evaluation.
     */
    virtual bool evaluate() const = 0;

    /**
     * @brief If evaluation is performed, the result will be stored here for future use.
     */
    mutable std::optional<bool> m_supported;
};

}  // namespace intel_npu
