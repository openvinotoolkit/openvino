// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "segment.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

class SurroundingSegmentsInserter {
public:
    virtual ~SurroundingSegmentsInserter() = default;
    virtual void insert_surrounding_segments(std::vector<Segment>& pwls) const = 0;
};

class SurroundingSegmentInserterCommon : public SurroundingSegmentsInserter {
public:
    SurroundingSegmentInserterCommon(double min_value, double max_value);

    void insert_surrounding_segments(std::vector<Segment>& pwls) const override;

private:
    double m_min_value;
    double m_max_value;
};

class SurroundingSegmentInserterPower : public SurroundingSegmentsInserter {
public:
    SurroundingSegmentInserterPower(double exponent);
    void insert_surrounding_segments(std::vector<Segment>& pwls) const override;

private:
    double m_exponent;
};

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov