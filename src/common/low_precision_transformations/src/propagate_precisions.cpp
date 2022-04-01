// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/propagate_precisions.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <low_precision/create_attribute.hpp>
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/propagate_through_precision_preserved.hpp"
#include "low_precision/propagate_to_input.hpp"
#include "itt.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

ngraph::pass::low_precision::PropagatePrecisions::PropagatePrecisions(const AttributeParameters& params) : params(params) {}

bool ngraph::pass::low_precision::PropagatePrecisions::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(PropagatePrecisions);
    ngraph::pass::Manager manager;
    manager.set_per_pass_validation(false);
    std::shared_ptr<ngraph::pass::GraphRewrite> precisionsPropagation = manager.register_pass<ngraph::pass::GraphRewrite>();
    precisionsPropagation->add_matcher<low_precision::CreateAttribute<PrecisionsAttribute, opset1::FakeQuantize>>(params, AttributeSource::OutputPort);
    precisionsPropagation->add_matcher<low_precision::PropagateThroughPrecisionPreserved<PrecisionsAttribute>>(params.defaultPrecisions);
    precisionsPropagation->add_matcher<low_precision::PropagateToInput<PrecisionsAttribute>>(params.defaultPrecisions);
    manager.run_passes(f);
    return false;
}
