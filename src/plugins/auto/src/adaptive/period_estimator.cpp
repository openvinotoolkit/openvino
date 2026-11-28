// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "period_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ov {
namespace auto_plugin {

namespace {
double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}
}  // namespace

std::optional<double> PeriodEstimator::median_ms() const {
    if (m_samples.size() < m_config.min_samples) {
        return std::nullopt;
    }
    return median_of({m_samples.begin(), m_samples.end()});
}

void PeriodEstimator::on_change(Time at, bool precise, unsigned max_skipped) {
    if (m_last && m_last_precise && precise) {
        const double gap = std::chrono::duration<double, std::milli>(at - *m_last).count();
        const auto known = median_ms();
        std::optional<double> sample = gap;
        if (known) {
            sample.reset();
            const double count = std::round(gap / *known);
            if (count >= 1 && count <= static_cast<double>(max_skipped)) {
                const double each = gap / count;
                if (std::abs(each - *known) <= m_config.tolerance * *known) {
                    sample = each;
                }
            }
        }
        const bool in_range = sample && *sample >= static_cast<double>(m_config.min_period.count()) &&
                              *sample <= static_cast<double>(m_config.max_period.count());
        if (in_range) {
            m_samples.push_back(*sample);
            if (m_samples.size() > m_config.max_samples) {
                m_samples.pop_front();
            }
            m_rejected = 0;
        } else if (known && ++m_rejected > m_config.max_rejected) {
            m_samples.clear();
            m_rejected = 0;
        }
    }
    m_last = at;
    m_last_precise = precise;
    m_lost = false;
}

void PeriodEstimator::on_lost() {
    m_lost = true;
    m_last_precise = false;
}

void PeriodEstimator::reset() {
    m_samples.clear();
    m_last.reset();
    m_last_precise = false;
    m_lost = false;
    m_rejected = 0;
}

bool PeriodEstimator::reliable() const {
    const auto median = median_ms();
    if (m_lost || !median) {
        return false;
    }
    double deviation = 0;
    for (const auto sample : m_samples) {
        deviation += std::abs(sample - *median);
    }
    deviation /= static_cast<double>(m_samples.size());
    return deviation <= m_config.max_spread * *median;
}

std::optional<std::chrono::milliseconds> PeriodEstimator::period() const {
    if (!reliable()) {
        return std::nullopt;
    }
    return std::chrono::milliseconds(static_cast<long long>(std::llround(*median_ms())));
}

}  // namespace auto_plugin
}  // namespace ov
