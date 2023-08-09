// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_useless_convert_like.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
//#include <ngraph/pattern/op/wrap_type.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

ov::pass::RemoveUselessConvertLike::RemoveUselessConvertLike() {
    MATCHER_SCOPE(RemoveUselessConvertLike);

    const auto const_pattern = pattern::any_input();
    const auto convert_like_pattern =
        pattern::wrap_type<ov::op::v1::ConvertLike>({const_pattern, pattern::any_input()});

    const matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto convert_like_node = std::dynamic_pointer_cast<ov::op::v1::ConvertLike>(m.get_match_root());
        auto const_node = convert_like_node->input_value(0).get_node_shared_ptr();
        if (!convert_like_node || !const_node)
            return false;

        if (convert_like_node->input(1).get_element_type() == const_node->output(0).get_element_type()) {
            copy_runtime_info(convert_like_node, const_node);
            const_node->set_friendly_name(convert_like_node->get_friendly_name());
            replace_node(convert_like_node, const_node);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(convert_like_pattern, matcher_name);
    register_matcher(m, callback);
}
