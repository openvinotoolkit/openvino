// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/align_quantization_intervals.hpp"
#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include "low_precision/create_attribute.hpp"
#include "low_precision/propagate_through_precision_preserved.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"

using namespace ov;
using namespace ov::pass::low_precision;

NGRAPH_RTTI_DEFINITION(ov::pass::low_precision::AlignQuantizationIntervals, "AlignQuantizationIntervals", 0);

bool ov::pass::low_precision::AlignQuantizationIntervals::run_on_function(std::shared_ptr<ov::Function> f) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    std::shared_ptr<ov::pass::GraphRewrite> intervalsAlignment = manager.register_pass<ov::pass::GraphRewrite>();
    intervalsAlignment->add_matcher<low_precision::CreateAttribute<IntervalsAlignmentAttributePtr, opset1::FakeQuantize>>();
    intervalsAlignment->add_matcher<low_precision::PropagateThroughPrecisionPreserved<IntervalsAlignmentAttribute>>();
    manager.run_passes(f);
    return false;
}
