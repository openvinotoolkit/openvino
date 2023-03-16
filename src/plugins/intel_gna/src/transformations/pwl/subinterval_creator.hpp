// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

// Represents subinterval for which segments will be generated.
class FunctionSubinterval {
public:
    FunctionSubinterval(bool negative, double lower_bound, double upper_bound);

    bool is_negative() const;
    double get_lower_bound() const;
    double get_upper_bound() const;

private:
    bool m_negative;
    double m_lower_bound;
    double m_upper_bound;
};

class FunctionSplitInfo {
public:
    virtual ~FunctionSplitInfo() = default;
    virtual bool should_split(double lower_bound, double upper_bound) const = 0;
    virtual double get_split_value() const = 0;
};

class FunctionSubintervalNegationInfo {
public:
    virtual ~FunctionSubintervalNegationInfo() = default;
    virtual bool is_negative(double lower_bound, double upper_bound) const = 0;
};

class SubintervalsCreator {
public:
    SubintervalsCreator(FunctionSplitInfo& split_checker, FunctionSubintervalNegationInfo& negation_checker);
    std::vector<FunctionSubinterval> generate(double lower_bound, double upper_bound) const;

private:
    const FunctionSplitInfo& m_split_checker;
    const FunctionSubintervalNegationInfo& m_negation_checker;
};

// Definition of supporting classes for function split info and function negation info.
class FunctionSplitInfoBasedOnBreakBoundValue : public FunctionSplitInfo {
public:
    FunctionSplitInfoBasedOnBreakBoundValue(double break_bound_value);

    bool should_split(double lower_bound, double upper_bound) const override;
    double get_split_value() const override;

private:
    double m_break_bound_value;
};

class FunctionSplitInfoNever : public FunctionSplitInfo {
public:
    bool should_split(double lower_bound, double upper_bound) const override;
    double get_split_value() const override;
};

class NegationInfoWhenUpperBoundEqualBreakBound : public FunctionSubintervalNegationInfo {
public:
    NegationInfoWhenUpperBoundEqualBreakBound(double break_bound_value);

    bool is_negative(double lower_bound, double upper_bound) const override;

private:
    double m_break_bound_value;
};

class FunctionSubintervalNegationInfoAlwaysTrue : public FunctionSubintervalNegationInfo {
public:
    bool is_negative(double lower_bound, double upper_bound) const override;
};

class FunctionSubintervalNegationInfoAlwaysFalse : public FunctionSubintervalNegationInfo {
public:
    bool is_negative(double lower_bound, double upper_bound) const override;
};

class FunctionSubintervalNegationInfoPower : public FunctionSubintervalNegationInfo {
public:
    FunctionSubintervalNegationInfoPower(double exponent, double break_bound_value);

    bool is_negative(double lower_bound, double upper_bound) const override;

private:
    double m_exponent;
    double m_break_bound_value;
};

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov