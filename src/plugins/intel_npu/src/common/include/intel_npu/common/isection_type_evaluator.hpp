// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "intel_npu/common/isection.hpp"

namespace intel_npu {

/**
 * @brief Abstract class that standardizes the evaluation of section types support in a lazy manner.
 */
class ISectionTypeEvaluator {
public:
    ISectionTypeEvaluator(const SectionType section_type);

    virtual ~ISectionTypeEvaluator() = default;

    SectionType get_section_type() const;

    /**
     * @brief Checks whether or not the NPU plugin supports the section type.
     * @details The evaluation depends on the implementation of the "evaluate" method. After evaluation, the
     * result is stored in "m_supported" for future use.
     */
    bool get_result() const;

    /**
     * @brief Tells whether or not the section type has already been evaluated.
     */
    bool evaluated() const;

private:
    /**
     * @brief Checks whether or not the NPU plugin supports the section type. This function will be called at most once
     * during execution, in which case the result will be stored for future use.
     * @note This method will typically be called only if the corresponding CRE token is found during CRE evaluation.
     */
    virtual bool evaluate() const = 0;

    SectionType m_section_type;
    /**
     * @brief If evaluation is performed, the result will be stored here for future use.
     */
    mutable std::optional<bool> m_supported;
};

}  // namespace intel_npu
