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

ngraph::pass::low_precision::PropagatePrecisions::PropagatePrecisions() {
    add_matcher<low_precision::CreateAttribute<PrecisionsAttributePtr, opset1::FakeQuantize>>(AttributeSource::OutputPort);
    add_matcher<low_precision::PropagateThroughPrecisionPreserved<PrecisionsAttribute>>();
    add_matcher<low_precision::PropagateToInput<PrecisionsAttribute>>();
}
