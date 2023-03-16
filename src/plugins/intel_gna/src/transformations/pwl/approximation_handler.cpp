// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "approximation_handler.hpp"

#include <algorithm>

#include "base_segment_precision_validator.hpp"
#include "common/numerical_utils.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gna {
using namespace common;
namespace pass {
namespace pwl {

ApproximationHandler::ApproximationHandler(
    const std::vector<std::shared_ptr<ActivationRepresentationConfiguration>>& configurations)
    : m_configurations(configurations) {
    auto input = ov::pass::pattern::any_input();

    m_fake_quantize_pattern = ov::pass::pattern::wrap_type<ov::opset11::FakeQuantize>({ov::pass::pattern::any_input(),
                                                                                       ov::pass::pattern::any_input(),
                                                                                       ov::pass::pattern::any_input(),
                                                                                       ov::pass::pattern::any_input(),
                                                                                       ov::pass::pattern::any_input()});

    auto activation_input_pattern =
        std::make_shared<ov::pass::pattern::op::Or>(ngraph::OutputVector{m_fake_quantize_pattern, input});

    for (const auto& configuration : m_configurations) {
        m_activations_pattern.push_back(configuration->get_pattern_node(activation_input_pattern));
    }

    m_pattern = std::make_shared<ov::pass::pattern::op::Or>(ov::as_output_vector(m_activations_pattern));
}

std::shared_ptr<ov::Node> ApproximationHandler::get_pattern() const {
    return m_pattern;
}

std::shared_ptr<SegmentsGenerator> ApproximationHandler::create_generator(
    const ov::pass::pattern::PatternValueMap& patterh_to_output) {
    std::shared_ptr<ov::Node> node;
    auto node_iter = patterh_to_output.end();

    for (const auto& pattern : m_activations_pattern) {
        node_iter = patterh_to_output.find(pattern);
        if (node_iter != patterh_to_output.end()) {
            node = node_iter->second.get_node_shared_ptr();
            break;
        }
    }

    if (nullptr == node) {
        return nullptr;
    }

    std::shared_ptr<ov::Node> fake_quantize;
    auto fake_quntize_iter = patterh_to_output.find(m_fake_quantize_pattern);
    if (fake_quntize_iter != patterh_to_output.end()) {
        fake_quantize = fake_quntize_iter->second.get_node_shared_ptr();
    }

    for (const auto& configuration : m_configurations) {
        auto representation = configuration->create_representation(node, fake_quantize);
        if (nullptr != representation) {
            return std::make_shared<SegmentsGeneratorImpl>(node, representation);
        }
    }
    return nullptr;
}

void ApproximationHandlerBuilder::add_configuration(
    std::shared_ptr<ActivationRepresentationConfiguration> configuration) {
    m_configurations.push_back(std::move(configuration));
}

std::shared_ptr<ApproximationHandler> ApproximationHandlerBuilder::build() {
    return std::make_shared<ApproximationHandler>(m_configurations);
}

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov