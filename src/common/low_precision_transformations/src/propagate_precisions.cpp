// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/propagate_precisions.hpp"

#include <memory>

#include "openvino/opsets/opset1.hpp"
#include "low_precision/create_attribute.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/propagate_through_precision_preserved.hpp"
#include "low_precision/propagate_to_input.hpp"
#include "itt.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;
using namespace ov::pass::low_precision;

ov::pass::low_precision::PropagatePrecisions::PropagatePrecisions(const AttributeParameters& params) : params(params) {}

bool ov::pass::low_precision::PropagatePrecisions::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(PropagatePrecisions);
    ov::pass::Manager manager("LPT:PropagatePrecisions");
    manager.set_per_pass_validation(false);
    std::shared_ptr<ov::pass::GraphRewrite> precisionsPropagation = manager.register_pass<ov::pass::GraphRewrite>();
    precisionsPropagation->add_matcher<low_precision::CreateAttribute<PrecisionsAttribute, opset1::FakeQuantize>>(params, AttributeSource::OutputPort);
    precisionsPropagation->add_matcher<low_precision::PropagateThroughPrecisionPreserved<PrecisionsAttribute>>(params.defaultPrecisions);
    precisionsPropagation->add_matcher<low_precision::PropagateToInput<PrecisionsAttribute>>(params.defaultPrecisions);
    manager.run_passes(f);
    return false;
}
