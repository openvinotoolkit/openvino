// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/align_quantization_intervals.hpp"
#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include "low_precision/create_attribute.hpp"
#include "low_precision/propagate_through_precision_preserved.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::AlignQuantizationIntervals, "AlignQuantizationIntervals", 0);

bool ngraph::pass::low_precision::AlignQuantizationIntervals::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    ngraph::pass::Manager manager;
    manager.set_per_pass_validation(false);
    std::shared_ptr<ngraph::pass::GraphRewrite> intervalsAlignment = manager.register_pass<ngraph::pass::GraphRewrite>();
    intervalsAlignment->add_matcher<low_precision::CreateAttribute<IntervalsAlignmentAttribute, opset1::FakeQuantize>>();
    intervalsAlignment->add_matcher<low_precision::PropagateThroughPrecisionPreserved<IntervalsAlignmentAttribute>>();
    manager.run_passes(f);
    return false;
}
