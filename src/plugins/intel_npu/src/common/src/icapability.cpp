// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/icapability.hpp"

namespace intel_npu {

ICapability::ICapability(const CRE::Token token) : m_token(token) {}

CRE::Token ICapability::get_token() const {
    return m_token;
}

bool ICapability::check_support() const {
    if (m_supported.has_value()) {
        return m_supported.value();
    }

    m_supported = lazy_check_support();
    return m_supported.value();
}

}  // namespace intel_npu
