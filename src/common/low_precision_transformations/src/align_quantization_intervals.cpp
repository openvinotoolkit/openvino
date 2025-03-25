// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/align_quantization_intervals.hpp"
#include <memory>
#include "openvino/opsets/opset1.hpp"
#include "low_precision/create_attribute.hpp"
#include "low_precision/propagate_through_precision_preserved.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/attribute_parameters.hpp"
#include "itt.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;
using namespace ov::pass::low_precision;

ov::pass::low_precision::AlignQuantizationIntervals::AlignQuantizationIntervals(const std::vector<ov::element::Type>& defaultPrecisions)
    : defaultPrecisions(defaultPrecisions) {}

bool ov::pass::low_precision::AlignQuantizationIntervals::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(AlignQuantizationIntervals);
    ov::pass::Manager manager("LPT:AlignQuantizationIntervals");
    manager.set_per_pass_validation(false);
    std::shared_ptr<ov::pass::GraphRewrite> intervalsAlignment = manager.register_pass<ov::pass::GraphRewrite>();
    intervalsAlignment->add_matcher<low_precision::CreateAttribute<IntervalsAlignmentAttribute, opset1::FakeQuantize>>(
        AttributeParameters(ov::element::f32, defaultPrecisions));
    intervalsAlignment->add_matcher<low_precision::PropagateThroughPrecisionPreserved<IntervalsAlignmentAttribute>>(defaultPrecisions);
    manager.run_passes(f);
    return false;
}
