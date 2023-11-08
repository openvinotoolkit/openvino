// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/enable_shapeof_constant_folding.hpp"

#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

ov::pass::EnableShapeOfConstantFolding::EnableShapeOfConstantFolding() {
    auto shape_of = pattern::wrap_type<op::util::ShapeOfBase>([=](const Output<Node>& output) {
        const auto& shape = output.get_partial_shape();
        return shape.is_dynamic() || shape_size(shape.get_shape()) != 1;
    });

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        enable_constant_folding(m.get_match_root());
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(shape_of, "EnableShapeOfConstantFolding");
    this->register_matcher(m, callback);
}
