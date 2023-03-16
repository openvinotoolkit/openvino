// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subinterval_creator.hpp"

#include <cmath>

#include "common/numerical_utils.hpp"

namespace ov {
namespace intel_gna {
using namespace common;

namespace pass {
namespace pwl {

FunctionSubinterval::FunctionSubinterval(bool negative, double lower_bound, double upper_bound)
    : m_negative(negative),
      m_lower_bound(lower_bound),
      m_upper_bound(upper_bound) {}

bool FunctionSubinterval::is_negative() const {
    return m_negative;
}

double FunctionSubinterval::get_lower_bound() const {
    return m_lower_bound;
}

double FunctionSubinterval::get_upper_bound() const {
    return m_upper_bound;
}

FunctionSplitInfoBasedOnBreakBoundValue::FunctionSplitInfoBasedOnBreakBoundValue(double break_bound_value)
    : m_break_bound_value(break_bound_value) {}

bool FunctionSplitInfoBasedOnBreakBoundValue::should_split(double lower_bound, double upper_bound) const {
    if (lower_bound > upper_bound) {
        return false;
    }

    return lower_bound < m_break_bound_value && upper_bound > m_break_bound_value;
}

double FunctionSplitInfoBasedOnBreakBoundValue::get_split_value() const {
    return m_break_bound_value;
}

bool FunctionSplitInfoNever::should_split(double lower_bound, double upper_bound) const {
    return false;
}

double FunctionSplitInfoNever::get_split_value() const {
    return 0.0;
}

NegationInfoWhenUpperBoundEqualBreakBound::NegationInfoWhenUpperBoundEqualBreakBound(double break_bound_value)
    : m_break_bound_value(break_bound_value) {}

bool NegationInfoWhenUpperBoundEqualBreakBound::is_negative(double lower_bound, double upper_bound) const {
    return AreFpEq(upper_bound, m_break_bound_value);
}

bool FunctionSubintervalNegationInfoAlwaysTrue::is_negative(double lower_bound, double upper_bound) const {
    return true;
}

bool FunctionSubintervalNegationInfoAlwaysFalse::is_negative(double lower_bound, double upper_bound) const {
    return false;
}

FunctionSubintervalNegationInfoPower::FunctionSubintervalNegationInfoPower(double exponent, double break_bound_value)
    : m_exponent(exponent),
      m_break_bound_value(break_bound_value) {}

bool FunctionSubintervalNegationInfoPower::is_negative(double lower_bound, double upper_bound) const {
    return AreFpEq(std::fmod(m_exponent, 1.0), m_break_bound_value);
}

SubintervalsCreator::SubintervalsCreator(FunctionSplitInfo& split_checker,
                                         FunctionSubintervalNegationInfo& negation_checker)
    : m_split_checker(split_checker),
      m_negation_checker(negation_checker) {}

std::vector<FunctionSubinterval> SubintervalsCreator::generate(double lower_bound, double upper_bound) const {
    std::vector<FunctionSubinterval> subintervals;
    if (m_split_checker.should_split(lower_bound, upper_bound)) {
        auto split_value = m_split_checker.get_split_value();
        subintervals.emplace_back(m_negation_checker.is_negative(lower_bound, split_value), lower_bound, split_value);
        subintervals.emplace_back(m_negation_checker.is_negative(split_value, upper_bound), split_value, upper_bound);
    } else {
        subintervals.emplace_back(m_negation_checker.is_negative(lower_bound, upper_bound), lower_bound, upper_bound);
    }
    return subintervals;
}
}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov