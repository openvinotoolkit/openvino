// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "activation_representation.hpp"
#include "activation_representation_configuration.hpp"
#include "boundaries_fq_hanlder.hpp"
#include "function.hpp"
#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "segment.hpp"
#include "segment_generator.hpp"
#include "subinterval_creator.hpp"
#include "surrounding_segments_inserter.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

class ApproximationHandler {
public:
    ApproximationHandler(const std::vector<std::shared_ptr<ActivationRepresentationConfiguration>>& configurations);

    std::shared_ptr<ov::Node> get_pattern() const;

    std::shared_ptr<SegmentsGenerator> create_generator(const ov::pass::pattern::PatternValueMap& patterh_to_output);

private:
    std::vector<std::shared_ptr<ActivationRepresentationConfiguration>> m_configurations;
    ov::NodeVector m_activations_pattern;
    std::shared_ptr<ov::Node> m_fake_quantize_pattern;
    std::shared_ptr<ov::Node> m_pattern;
};

class ApproximationHandlerBuilder {
public:
    void add_configuration(std::shared_ptr<ActivationRepresentationConfiguration> configuration);
    std::shared_ptr<ApproximationHandler> build();

private:
    std::vector<std::shared_ptr<ActivationRepresentationConfiguration>> m_configurations;
};
}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov