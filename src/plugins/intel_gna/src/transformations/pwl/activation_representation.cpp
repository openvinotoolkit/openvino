// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_representation.hpp"

#include "ie_common.h"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

PrecalculatedActivationSegments::PrecalculatedActivationSegments(const std::vector<Segment>& segments)
    : m_segments(segments) {}

std::vector<Segment> PrecalculatedActivationSegments::get_precalculated_segments() const {
    return m_segments;
}

ActivationDataForCalculations::ActivationDataForCalculations(
    const std::shared_ptr<Function>& function,
    const std::vector<FunctionSubinterval>& subintervals,
    const std::shared_ptr<SurroundingSegmentsInserter>& surrounding_segments_inserter,
    double precision,
    size_t max_iterations)
    : m_function(function),
      m_subintervals(subintervals),
      m_surrounding_segments_inserter(surrounding_segments_inserter),
      m_precision(precision),
      m_max_iterations(max_iterations) {}

std::vector<FunctionSubinterval> ActivationDataForCalculations::get_subintervals() const {
    return m_subintervals;
}

const Function& ActivationDataForCalculations::get_function() const {
    return *m_function;
}

double ActivationDataForCalculations::get_allowed_error_percentage() const {
    return m_precision;
}

std::shared_ptr<SurroundingSegmentsInserter> ActivationDataForCalculations::get_surrounding_segments_inserter() const {
    return m_surrounding_segments_inserter;
}

size_t ActivationDataForCalculations::get_max_iterations() const {
    return m_max_iterations;
}

ActivationRepresentationImpl::ActivationRepresentationImpl(
    const std::string& name,
    const std::shared_ptr<PrecalculatedActivationSegments>& segments)
    : m_name(name),
      m_precalculated_segments(segments) {}

ActivationRepresentationImpl::ActivationRepresentationImpl(
    const std::string& name,
    const std::shared_ptr<ActivationDataForCalculations>& data_for_calculation)
    : m_name(name),
      m_data_for_calculation(data_for_calculation) {}

std::string ActivationRepresentationImpl::get_name() const {
    return m_name;
}

std::shared_ptr<PrecalculatedActivationSegments> ActivationRepresentationImpl::get_precalculated_segments() const {
    return m_precalculated_segments;
}

std::shared_ptr<ActivationDataForCalculations> ActivationRepresentationImpl::get_data_for_calculation() const {
    return m_data_for_calculation;
}

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov