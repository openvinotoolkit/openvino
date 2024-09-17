// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_shapeof.hpp"

#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/core/rt_info.hpp"
#include <memory>
#include <vector>

ov::intel_gpu::ConvertShapeOf1To3::ConvertShapeOf1To3() {
    auto shapeof1 = ov::pass::pattern::wrap_type<ov::op::v0::ShapeOf>();

    matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto shapeof1 = ov::as_type_ptr<ov::op::v0::ShapeOf>(m.get_match_root());
        if (!shapeof1) {
            return false;
        }

        auto new_shapeof3 = std::make_shared<ov::op::v3::ShapeOf>(shapeof1->input_value(0));
        new_shapeof3->set_friendly_name(shapeof1->get_friendly_name());
        ov::copy_runtime_info(shapeof1, new_shapeof3);
        ov::replace_node(shapeof1, new_shapeof3);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(shapeof1, "ConvertShapeOf1To3");
    register_matcher(m, callback);
}
