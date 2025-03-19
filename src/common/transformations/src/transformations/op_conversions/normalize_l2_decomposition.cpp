// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/normalize_l2_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::NormalizeL2Decomposition::NormalizeL2Decomposition() {
    MATCHER_SCOPE(NormalizeL2Decomposition);
    auto normalize_l2_pattern = ov::pass::pattern::wrap_type<ov::op::v0::NormalizeL2>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto normalize_l2 = ov::as_type_ptr<ov::op::v0::NormalizeL2>(m.get_match_root());

        if (!normalize_l2 || transformation_callback(normalize_l2)) {
            return false;
        }

        auto power = std::make_shared<ov::op::v1::Power>(
            normalize_l2->input_value(0),
            ov::op::v0::Constant::create(normalize_l2->get_input_element_type(0), Shape{}, {2.0}));
        auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(power, normalize_l2->input_value(1), true);

        std::shared_ptr<Node> eps_node;
        auto eps_const_node =
            ov::op::v0::Constant::create(normalize_l2->get_input_element_type(0), Shape{}, {normalize_l2->get_eps()});
        switch (normalize_l2->get_eps_mode()) {
        case op::EpsMode::ADD:
            eps_node = std::make_shared<ov::op::v1::Add>(reduce_sum, eps_const_node);
            break;
        case op::EpsMode::MAX:
            eps_node = std::make_shared<ov::op::v1::Maximum>(reduce_sum, eps_const_node);
            break;
        default:
            return false;
        }

        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(eps_node);
        auto div = std::make_shared<ov::op::v1::Divide>(normalize_l2->input_value(0), sqrt);

        div->set_friendly_name(normalize_l2->get_friendly_name());
        ov::copy_runtime_info(normalize_l2, {power, reduce_sum, eps_node, sqrt, div});
        ov::replace_node(normalize_l2, div);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(normalize_l2_pattern, matcher_name);
    register_matcher(m, callback);
}
