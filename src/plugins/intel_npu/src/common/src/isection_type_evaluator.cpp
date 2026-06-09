// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/isection_type_evaluator.hpp"

namespace intel_npu {

ISectionTypeEvaluator::ISectionTypeEvaluator(const CRE::Token token) : m_token(token) {}

CRE::Token ISectionTypeEvaluator::get_token() const {
    return m_token;
}

bool ISectionTypeEvaluator::check_support() const {
    if (m_supported.has_value()) {
        return m_supported.value();
    }

    m_supported = lazy_check_support();
    return m_supported.value();
}

}  // namespace intel_npu
