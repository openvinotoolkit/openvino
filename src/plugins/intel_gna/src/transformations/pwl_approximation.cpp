// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl_approximation.hpp"

#include <memory>
#include <ngraph/rt_info.hpp>
#include <openvino/cc/ngraph/itt.hpp>
#include <vector>

#include "common/graph_utils.hpp"
#include "log/log.hpp"
#include "openvino/opsets/opset11.hpp"
#include "ops/pwl.hpp"
#include "pwl/approximation_handler.hpp"
#include "pwl/configuration_exp.hpp"
#include "pwl/configuration_log.hpp"
#include "pwl/configuration_power.hpp"
#include "pwl/configuration_sigmoid.hpp"
#include "pwl/configuration_softsign.hpp"
#include "pwl/configuration_tanh.hpp"
#include "pwl/segment.hpp"

using namespace ov::intel_gna;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::pass::pwl;

NGRAPH_RTTI_DEFINITION(PWLApproximation, "PWLApproximation");

static bool transform_to_pwl(ov::pass::pattern::Matcher& matcher, const std::shared_ptr<SegmentsGenerator>& generator) {
    if (nullptr == generator) {
        return false;
    }

    auto segments = generator->generate_segments();

    if (segments.empty()) {
        return false;
    }

    std::vector<double> m(segments.size() - 1);
    std::vector<double> b(segments.size() - 1);
    std::vector<double> alpha(segments.size());
    for (size_t i = 0; i < segments.size() - 1; i++) {
        m[i] = segments[i].m;
        b[i] = segments[i].b;
        alpha[i] = segments[i].alpha;
    }

    auto node = generator->get_node();
    alpha[segments.size() - 1] = segments[segments.size() - 1].alpha;
    auto m_constant =
        std::make_shared<ov::opset11::Constant>(ov::element::Type_t::f64, ov::Shape{segments.size() - 1}, m);
    m_constant->set_friendly_name(node->get_friendly_name() + "/pwl_slope");
    auto b_constant =
        std::make_shared<ov::opset11::Constant>(ov::element::Type_t::f64, ov::Shape{segments.size() - 1}, b);
    b_constant->set_friendly_name(node->get_friendly_name() + "/pwl_offset");
    auto alpha_constant =
        std::make_shared<ov::opset11::Constant>(ov::element::Type_t::f64, ov::Shape{segments.size()}, alpha);
    alpha_constant->set_friendly_name(node->get_friendly_name() + "/pwl_alpha");

    // TODO in previous implementation in case FakeQuantize was found was used as first argument
    // otherwise node->input_value(0) was used. It seems node->input_value(0) can be used as well for both cases.
    auto pwl = std::make_shared<ov::intel_gna::op::Pwl>(node->input_value(0), m_constant, b_constant, alpha_constant);
    pwl->set_base_node(node);
    pwl->set_friendly_name(node->get_friendly_name());
    ov::copy_runtime_info(node, {pwl, m_constant, b_constant, alpha_constant});
    replace_node(node, pwl);
    return true;
}

PWLApproximation::PWLApproximation(const PWLApproximationMode& mode) {
    MATCHER_SCOPE(PWLApproximation);

    auto builder = std::make_shared<ApproximationHandlerBuilder>();

    builder->add_configuration(ConfigurationSigmoid::create_config(mode));
    builder->add_configuration(ConfigurationTanh::create_config(mode));
    builder->add_configuration(ConfigurationPower::create_config(mode));
    builder->add_configuration(ConfigurationSoftSign::create_config(mode));
    builder->add_configuration(ConfigurationExp::create_config(mode));
    builder->add_configuration(ConfigurationLog::create_config(mode));

    auto approximation_handler = builder->build();
    auto pattern = approximation_handler->get_pattern();

    auto callback = [approximation_handler](ov::pass::pattern::Matcher& matcher) -> bool {
        const auto& pattern_to_output = matcher.get_pattern_value_map();
        auto generator = approximation_handler->create_generator(pattern_to_output);
        return transform_to_pwl(matcher, generator);
    };

    auto macher = std::make_shared<ov::pass::pattern::Matcher>(pattern, matcher_name);

    register_matcher(macher, callback);
}
