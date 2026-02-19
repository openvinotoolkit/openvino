// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "intel_npu/common/cre.hpp"

namespace intel_npu {

/**
 * @brief Interface that standardizes the evaluation of an NPU plugin capability.
 */
class ICapability {
public:
    ICapability(const CRE::Token token);

    virtual ~ICapability() = default;

    CRE::Token get_token() const;

    /**
     * @brief Checks if the NPU plugin has the current capability.
     * @details The evaluation depends on the implementation of the "lazy_check_support" method. After evaluation, the
     * result is stored in "m_supported" for future use.
     */
    bool check_support() const;

    /**
     * @brief Checks if the NPU plugin has the current capability. This function will be called at most once during
     * execution, in which case the result will be stored for future use.
     * @note This method will typically be called only if the corresponding CRE token is found during CRE evaluation.
     */
    virtual bool lazy_check_support() const = 0;

private:
    CRE::Token m_token;
    /**
     * @brief If evaluation is performed, the result will be stored here for future use.
     */
    mutable std::optional<bool> m_supported;
};

}  // namespace intel_npu
