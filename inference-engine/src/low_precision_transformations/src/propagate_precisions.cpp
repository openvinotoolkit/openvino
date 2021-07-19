// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/propagate_precisions.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <low_precision/create_attribute.hpp>
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/propagate_through_precision_preserved.hpp"
#include "low_precision/propagate_to_input.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::PropagatePrecisions, "PropagatePrecisions", 0);

bool ngraph::pass::low_precision::PropagatePrecisions::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager;
    manager.set_per_pass_validation(false);
    std::shared_ptr<ngraph::pass::GraphRewrite> precisionsPropagation = manager.register_pass<ngraph::pass::GraphRewrite>();
    precisionsPropagation->add_matcher<low_precision::CreateAttribute<PrecisionsAttributePtr, opset1::FakeQuantize>>(AttributeSource::OutputPort);
    precisionsPropagation->add_matcher<low_precision::PropagateThroughPrecisionPreserved<PrecisionsAttribute>>();
    precisionsPropagation->add_matcher<low_precision::PropagateToInput<PrecisionsAttribute>>();
    manager.run_passes(f);
    return false;
}
