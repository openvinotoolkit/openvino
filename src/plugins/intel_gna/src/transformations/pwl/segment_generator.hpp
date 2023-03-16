// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <string>
#include <vector>

#include "activation_representation.hpp"
#include "function.hpp"
#include "openvino/core/node.hpp"
#include "segment.hpp"
#include "subinterval_creator.hpp"
#include "surrounding_segments_inserter.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

class SegmentsGenerator {
public:
    virtual ~SegmentsGenerator() = default;
    virtual std::vector<Segment> generate_segments() const = 0;
    virtual std::shared_ptr<ov::Node> get_node() const = 0;
};

class SegmentsGeneratorImpl : public SegmentsGenerator {
public:
    SegmentsGeneratorImpl(const std::shared_ptr<ov::Node>& node,
                          std::shared_ptr<ActivationRepresentation> representation);

    std::vector<Segment> generate_segments() const override;

    std::shared_ptr<ov::Node> get_node() const override;

private:
    std::pair<double, std::vector<Segment>> pivot_search(const Function& activation_function,
                                                         uint32_t segments_num,
                                                         double alpha_0,
                                                         double alpha_N,
                                                         bool negative,
                                                         size_t max_iterations) const;

    void negative_pwls(std::vector<Segment>& data) const;

    static constexpr int kMinSegmentsNum = 5;
    static constexpr const int kMaxSegmentsOverall = 128;
    // Leave two places for serrouding segments
    static constexpr const int kMaxSegmentsNum = kMaxSegmentsOverall - 2;
    static constexpr const int kSamplesNum = 500;
    static constexpr const double kThreshold = 0.1;

    std::shared_ptr<ov::Node> m_node;
    std::shared_ptr<ActivationRepresentation> m_representation;
};

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov