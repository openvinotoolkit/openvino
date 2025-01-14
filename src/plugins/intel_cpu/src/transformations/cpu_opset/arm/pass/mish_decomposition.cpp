// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mish_decomposition.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset4.hpp"

ov::intel_cpu::MishDecomposition::MishDecomposition() {
    auto mish = ov::pass::pattern::wrap_type<opset4::Mish>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto mish = ov::as_type_ptr<opset4::Mish>(m.get_match_root());
        if (!mish) {
            return false;
        }

        auto exp = std::make_shared<opset4::Exp>(mish->input_value(0));
        auto add = std::make_shared<opset4::Add>(
            exp,
            opset4::Constant::create(mish->get_output_element_type(0), ov::Shape{}, {1.0f}));
        auto log = std::make_shared<opset4::Log>(add);
        auto tanh = std::make_shared<opset4::Tanh>(log);
        auto mul = std::make_shared<opset4::Multiply>(mish->input_value(0), tanh);

        mul->set_friendly_name(mish->get_friendly_name());
        ov::copy_runtime_info(mish, {exp, add, log, tanh, mul});
        ov::replace_node(mish, mul);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mish, "MishDecomposition");
    register_matcher(m, callback);
}
