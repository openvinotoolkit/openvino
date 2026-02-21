// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "prim_layout_replacer.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

PrimLayoutReplacer::PrimLayoutReplacer() {
    auto op = ov::pass::pattern::wrap_type<ov::frontend::pytorch::PtFrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto node = ov::as_type_ptr<ov::frontend::pytorch::PtFrameworkNode>(m.get_match_root());
        if (!node || node->get_decoder()->get_op_type() != "prim::layout") {
            return false;
        }

        auto const_zero =
            std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
        const_zero->set_friendly_name(node->get_friendly_name());

        ov::replace_node(node, const_zero);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(op, "PrimLayoutReplacer");
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
