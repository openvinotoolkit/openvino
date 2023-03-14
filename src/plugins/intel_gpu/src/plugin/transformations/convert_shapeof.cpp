// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_shapeof.hpp"

#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <vector>

ov::intel_gpu::ConvertShapeOf1To3::ConvertShapeOf1To3() {
    auto shapeof1 = ov::pass::pattern::wrap_type<ov::opset1::ShapeOf>();

    matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto shapeof1 = std::dynamic_pointer_cast<ov::opset1::ShapeOf>(m.get_match_root());
        if (!shapeof1) {
            return false;
        }

        auto new_shapeof3 = std::make_shared<ov::opset3::ShapeOf>(shapeof1->input_value(0));
        new_shapeof3->set_friendly_name(shapeof1->get_friendly_name());
        ngraph::copy_runtime_info(shapeof1, new_shapeof3);
        ngraph::replace_node(shapeof1, new_shapeof3);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(shapeof1, "ConvertShapeOf1To3");
    register_matcher(m, callback);
}
