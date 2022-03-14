// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/common_optimizations/disable_shapeof_constant_folding.hpp>
#include <transformations/rt_info/disable_constant_folding.hpp>

ngraph::pass::DisableShapeOfConstantFolding::DisableShapeOfConstantFolding() {
    auto shape_of = pattern::wrap_type<opset2::ShapeOf, opset3::ShapeOf>([=](const Output<Node>& output) {
        const auto& shape = output.get_partial_shape();
        return shape.is_dynamic() || shape_size(shape.get_shape()) != 1;
    });

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        disable_constant_folding(m.get_match_root());
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(shape_of, "DisableShapeOfConstantFolding");
    this->register_matcher(m, callback);
}
