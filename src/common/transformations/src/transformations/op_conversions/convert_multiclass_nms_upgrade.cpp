// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_multiclass_nms_upgrade.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

ngraph::pass::ConvertMulticlassNms8ToMulticlassNms9::ConvertMulticlassNms8ToMulticlassNms9() {
    MATCHER_SCOPE(ConvertMulticlassNms8ToMulticlassNms9);

    auto nms_v8_pattern = pattern::wrap_type<opset8::MulticlassNms>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto nms_v8_node = std::dynamic_pointer_cast<opset8::MulticlassNms>(m.get_match_root());
        if (!nms_v8_node)
            return false;

        const auto new_args = nms_v8_node->input_values();
        // vector of new nGraph operations
        NodeVector new_ops;
        auto attrs = nms_v8_node->get_attrs();
        auto nms_v9_node = std::make_shared<opset9::MulticlassNms>(new_args.at(0), new_args.at(1), attrs);
        nms_v9_node->set_friendly_name(nms_v8_node->get_friendly_name());
        copy_runtime_info(nms_v8_node, nms_v9_node);
        replace_node(nms_v8_node, nms_v9_node);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(nms_v8_pattern, matcher_name);
    register_matcher(m, callback);
}
