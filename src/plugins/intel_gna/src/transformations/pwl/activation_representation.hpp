// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <string>
#include <vector>

#include "function.hpp"
#include "segment.hpp"
#include "subinterval_creator.hpp"
#include "surrounding_segments_inserter.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

class PrecalculatedActivationSegments {
public:
    PrecalculatedActivationSegments(const std::vector<Segment>& segments);

    std::vector<Segment> get_precalculated_segments() const;

private:
    std::vector<Segment> m_segments;
};

class ActivationDataForCalculations {
public:
    ActivationDataForCalculations(const std::shared_ptr<Function>& function,
                                  const std::vector<FunctionSubinterval>& subintervals,
                                  const std::shared_ptr<SurroundingSegmentsInserter>& surrounding_segments_inserter,
                                  double precision,
                                  size_t max_iterations);

    std::vector<FunctionSubinterval> get_subintervals() const;
    const Function& get_function() const;
    std::shared_ptr<SurroundingSegmentsInserter> get_surrounding_segments_inserter() const;
    double get_allowed_error_percentage() const;
    size_t get_max_iterations() const;

private:
    std::shared_ptr<Function> m_function;
    std::vector<FunctionSubinterval> m_subintervals;
    std::shared_ptr<SurroundingSegmentsInserter> m_surrounding_segments_inserter;
    double m_precision;
    size_t m_max_iterations;
};

class ActivationRepresentation {
public:
    virtual ~ActivationRepresentation() = default;
    virtual std::string get_name() const = 0;
    virtual std::shared_ptr<PrecalculatedActivationSegments> get_precalculated_segments() const = 0;
    virtual std::shared_ptr<ActivationDataForCalculations> get_data_for_calculation() const = 0;
};

class ActivationRepresentationImpl : public ActivationRepresentation {
public:
    ActivationRepresentationImpl(const std::string& name,
                                 const std::shared_ptr<PrecalculatedActivationSegments>& precalculated_segments);
    ActivationRepresentationImpl(const std::string& name,
                                 const std::shared_ptr<ActivationDataForCalculations>& data_for_calculation);

    std::string get_name() const override;
    std::shared_ptr<PrecalculatedActivationSegments> get_precalculated_segments() const override;
    std::shared_ptr<ActivationDataForCalculations> get_data_for_calculation() const override;

private:
    std::string m_name;

    std::shared_ptr<PrecalculatedActivationSegments> m_precalculated_segments;
    std::shared_ptr<ActivationDataForCalculations> m_data_for_calculation;
};

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov