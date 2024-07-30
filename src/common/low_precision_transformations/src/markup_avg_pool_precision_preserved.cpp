// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_avg_pool_precision_preserved.hpp"
#include <memory>
#include "openvino/opsets/opset1.hpp"
#include "low_precision/create_precisions_dependent_attribute.hpp"
#include "low_precision/rt_info/avg_pool_precision_preserved_attribute.hpp"
#include "low_precision/propagate_through_precision_preserved.hpp"
#include "low_precision/update_shared_precision_preserved.hpp"
#include "itt.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;

ov::pass::low_precision::MarkupAvgPoolPrecisionPreserved::MarkupAvgPoolPrecisionPreserved(const std::vector<ov::element::Type> defaultPrecisions)
    : defaultPrecisions(defaultPrecisions) {}

bool ov::pass::low_precision::MarkupAvgPoolPrecisionPreserved::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(MarkupAvgPoolPrecisionPreserved);
    ov::pass::Manager manager("LPT:MarkupAvgPoolPrecisionPreserved");
    manager.set_per_pass_validation(false);
    std::shared_ptr<ov::pass::GraphRewrite> markupAvgPoolPrecision = manager.register_pass<ov::pass::GraphRewrite>();
    markupAvgPoolPrecision->add_matcher<low_precision::CreatePrecisionsDependentAttribute<AvgPoolPrecisionPreservedAttribute, opset1::AvgPool>>();
    markupAvgPoolPrecision->add_matcher<low_precision::PropagateThroughPrecisionPreserved<AvgPoolPrecisionPreservedAttribute>>(defaultPrecisions);
    markupAvgPoolPrecision->add_matcher<low_precision::UpdateSharedPrecisionPreserved<AvgPoolPrecisionPreservedAttribute>>(defaultPrecisions);
    manager.run_passes(f);
    return false;
}
