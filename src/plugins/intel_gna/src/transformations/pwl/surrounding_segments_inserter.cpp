// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "surrounding_segments_inserter.hpp"

#include <cmath>

#include "common/numerical_utils.hpp"

namespace ov {
namespace intel_gna {
using namespace common;
namespace pass {
namespace pwl {

SurroundingSegmentInserterCommon::SurroundingSegmentInserterCommon(double min_value, double max_value)
    : m_min_value(min_value),
      m_max_value(max_value) {}

void SurroundingSegmentInserterCommon::insert_surrounding_segments(std::vector<Segment>& pwls) const {
    if (pwls.size() <= 2) {
        return;
    }

    if (pwls.front().beta < m_min_value) {
        pwls.front().alpha += (m_min_value - pwls.front().beta) / pwls.front().m;
    }

    pwls.insert(pwls.begin(), {0, std::max(pwls.front().beta, m_min_value), -std::numeric_limits<double>::infinity()});
    if (pwls.back().beta > m_max_value) {
        pwls.back().alpha += (m_max_value - pwls.back().beta) / pwls.at(pwls.size() - 2).m;
    }

    pwls.back().b = std::min(pwls.back().beta, m_max_value);
    pwls.push_back({0, 0, std::numeric_limits<double>::infinity()});
}

SurroundingSegmentInserterPower::SurroundingSegmentInserterPower(double exponent) : m_exponent(exponent) {}

void SurroundingSegmentInserterPower::insert_surrounding_segments(std::vector<Segment>& pwls) const {
    if (pwls.size() <= 2) {
        return;
    }

    pwls.insert(
        pwls.begin(),
        {0, pwls.front().beta, AreFpEq(fmod(m_exponent, 1.0), 0.0) ? -std::numeric_limits<double>::infinity() : 0});
    pwls.back().b = pwls.back().beta;
    pwls.push_back({0, 0, std::numeric_limits<double>::infinity()});
}
}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov