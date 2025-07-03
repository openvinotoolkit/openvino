// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mish_decomposition.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::intel_cpu::MishDecomposition::MishDecomposition() {
    auto mish = ov::pass::pattern::wrap_type<ov::op::v4::Mish>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto mish = ov::as_type_ptr<ov::op::v4::Mish>(m.get_match_root());
        if (!mish) {
            return false;
        }

        auto exp = std::make_shared<ov::op::v0::Exp>(mish->input_value(0));
        auto add = std::make_shared<ov::op::v1::Add>(
            exp,
            op::v0::Constant::create(mish->get_output_element_type(0), ov::Shape{}, {1.0F}));
        auto log = std::make_shared<ov::op::v0::Log>(add);
        auto tanh = std::make_shared<ov::op::v0::Tanh>(log);
        auto mul = std::make_shared<ov::op::v1::Multiply>(mish->input_value(0), tanh);

        mul->set_friendly_name(mish->get_friendly_name());
        ov::copy_runtime_info(mish, {exp, add, log, tanh, mul});
        ov::replace_node(mish, mul);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mish, "MishDecomposition");
    register_matcher(m, callback);
}
