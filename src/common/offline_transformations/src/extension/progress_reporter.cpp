// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension/progress_reporter.hpp"

#include <algorithm>

namespace ov {
namespace frontend {
ProgressCounter::ProgressCounter(unsigned int steps) : m_total_steps{steps} {}

float ProgressCounter::advance(unsigned int steps) {
    m_completed_steps = std::min(m_total_steps, m_completed_steps + steps);
    return current_progress();
}

ProgressCounter& ProgressCounter::operator++() {
    advance();
    return *this;
}

ProgressCounter& ProgressCounter::operator++(int) {
    return this->operator++();
}

float ProgressCounter::current_progress() const {
    return m_completed_steps / static_cast<float>(m_total_steps);
}

ProgressReporterExtension::ProgressReporterExtension() {}
}  // namespace frontend
}  // namespace ov
