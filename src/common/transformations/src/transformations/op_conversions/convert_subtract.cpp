// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_subtract.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/dequantization_node.hpp"

using namespace ov;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace ov::pass {

namespace {
bool convert_subtract(const std::shared_ptr<Node>& node) {
    auto sub = ov::as_type_ptr<v1::Subtract>(node);
    if (!sub) {
        return false;
    }

    if (ov::is_dequantization_node(sub)) {
        return false;
    }

    if (!sub->get_input_element_type(1).is_signed()) {
        return false;
    }

    std::shared_ptr<Node> neg =
        std::make_shared<v1::Multiply>(sub->input_value(1),
                                       v0::Constant::create(sub->get_input_element_type(1), Shape{}, {-1}));
    NodeVector new_nodes;
    if (auto constant = ov::util::get_constant_from_source(neg)) {
        neg = constant;
    } else {
        new_nodes.push_back(neg);
    }

    auto add = std::make_shared<v1::Add>(sub->input_value(0), neg);
    new_nodes.push_back(add);

    add->set_friendly_name(sub->get_friendly_name());
    copy_runtime_info(sub, new_nodes);
    replace_node(sub, add);

    return true;
}
}  // namespace

ConvertSubtract::ConvertSubtract() {
    MATCHER_SCOPE(ConvertSubtract);
    auto sub = pattern::wrap_type<v1::Subtract>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        return convert_subtract(node);
    };

    auto m = std::make_shared<pattern::Matcher>(sub, matcher_name);
    this->register_matcher(m, callback);
}

ConvertSubtractWithConstant::ConvertSubtractWithConstant() {
    MATCHER_SCOPE(ConvertSubtractWithConstant);
    auto sub = pattern::wrap_type<v1::Subtract>({pattern::any_input(), pattern::wrap_type<v0::Constant>()});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        return convert_subtract(node);
    };

    auto m = std::make_shared<pattern::Matcher>(sub, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
