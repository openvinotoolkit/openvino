// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_capabilities.hpp"

#include <algorithm>

bool MockCapability_1::lazy_check_support() const {
    return m_section->get_value() < VALID_THRESHOLD;
}

bool MockCapability_2::lazy_check_support() const {
    const auto vals = m_section->get_values();
    return std::all_of(vals.begin(), vals.end(), [](double v) {
        return v < VALID_THRESHOLD;
    });
}

bool MockCapability_3::lazy_check_support() const {
    auto [val, vals] = m_section->get_values();
    MockCapability_1 s1_cap(std::make_shared<MockSection_1>(val));
    MockCapability_2 s2_cap(std::make_shared<MockSection_2>(vals));
    return s1_cap.check_support() && s2_cap.check_support();
}

bool DriverCapability::lazy_check_support() const {
    return m_driver.supports_section(m_type);
}
