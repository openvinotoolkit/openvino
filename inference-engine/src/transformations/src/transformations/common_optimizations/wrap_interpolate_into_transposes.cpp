// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/wrap_interpolate_into_transposes.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

namespace {
bool transformation_is_applicable(const std::shared_ptr<ngraph::opset8::Interpolate>& interpolate) {
    if (interpolate->get_input_partial_shape(0).rank().is_dynamic()) return false;
}
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::WrapInterpolateIntoTransposes, "WrapInterpolateIntoTransposes", 0);

ngraph::pass::WrapInterpolateIntoTransposes::WrapInterpolateIntoTransposes() {
    MATCHER_SCOPE(WrapInterpolateIntoTransposes);
    auto interpolate_pattern = ngraph::pattern::wrap_type<ngraph::opset8::Interpolate>();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto interpolate = std::dynamic_pointer_cast<opset8::Interpolate>(m.get_match_root());
        if (!interpolate || !transformation_is_applicable(interpolate)) return false;

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(interpolate_pattern, matcher_name);
    register_matcher(m, callback);
}
