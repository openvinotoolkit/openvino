// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_upgrade.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertGather1ToGather7, "ConvertGather1ToGather7", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertGather7ToGather8, "ConvertGather7ToGather8", 0);

ngraph::pass::ConvertGather1ToGather7::ConvertGather1ToGather7() {
    MATCHER_SCOPE(ConvertGather1ToGather7);

    auto gather_v1 = pattern::wrap_type<ngraph::opset1::Gather>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto gather_v1_node = m.get_match_root();
        if (!gather_v1_node)
            return false;

        auto data_input = gather_v1_node->input_value(0);
        auto indices_input = gather_v1_node->input_value(1);
        auto axis_input = gather_v1_node->input_value(2);

        auto gather_v7  = std::make_shared<ngraph::opset7::Gather>(data_input, indices_input, axis_input, 0);
        gather_v7->set_friendly_name(gather_v1_node->get_friendly_name());
        ngraph::copy_runtime_info(gather_v1_node, gather_v7);
        ngraph::replace_node(gather_v1_node, gather_v7);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(gather_v1, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::ConvertGather7ToGather8::ConvertGather7ToGather8() {
    MATCHER_SCOPE(ConvertGather7ToGather8);

    auto gather_v7 = pattern::wrap_type<ngraph::opset7::Gather>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto gather_v1_node = m.get_match_root();
        if (!gather_v1_node)
            return false;

        auto data_input = gather_v1_node->input_value(0);
        auto indices_input = gather_v1_node->input_value(1);
        auto axis_input = gather_v1_node->input_value(2);

        auto gather_v8  = std::make_shared<ngraph::opset8::Gather>(data_input, indices_input, axis_input, 0);
        gather_v8->set_friendly_name(gather_v1_node->get_friendly_name());
        ngraph::copy_runtime_info(gather_v1_node, gather_v8);
        ngraph::replace_node(gather_v1_node, gather_v8);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(gather_v7, matcher_name);
    register_matcher(m, callback);
}
