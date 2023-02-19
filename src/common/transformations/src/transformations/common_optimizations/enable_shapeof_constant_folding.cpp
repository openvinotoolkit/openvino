// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pattern/op/wrap_type.hpp>
#include <openvino/op/shape_of.hpp>
#include <transformations/common_optimizations/enable_shapeof_constant_folding.hpp>
#include <transformations/rt_info/disable_constant_folding.hpp>

ov::pass::EnableShapeOfConstantFolding::EnableShapeOfConstantFolding() {
    auto shape_of = pattern::wrap_type<op::v0::ShapeOf, op::v3::ShapeOf>([=](const Output<Node>& output) {
        const auto& shape = output.get_partial_shape();
        return shape.is_dynamic() || shape_size(shape.get_shape()) != 1;
    });

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        enable_constant_folding(m.get_match_root());
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(shape_of, "EnableShapeOfConstantFolding");
    this->register_matcher(m, callback);
}
