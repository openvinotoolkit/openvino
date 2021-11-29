// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_v7_to_gather_v1.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertGather7ToGather1, "ConvertGather7ToGather1", 0);

ngraph::pass::ConvertGather7ToGather1::ConvertGather7ToGather1() {
    MATCHER_SCOPE(ConvertGather7ToGather1);

    auto gather_v7 = pattern::wrap_type<ngraph::opset7::Gather>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto gather_v7_node = std::dynamic_pointer_cast<ngraph::opset7::Gather>(m.get_match_root());
        if (!gather_v7_node)
            return false;

        if (gather_v7_node->get_batch_dims() != 0)
            return false;

        auto data_input = gather_v7_node->input_value(0);
        auto indices_input = gather_v7_node->input_value(1);
        auto axis_input = gather_v7_node->input_value(2);

        auto gather_v1 = std::make_shared<ngraph::opset1::Gather>(data_input, indices_input, axis_input);
        gather_v1->set_friendly_name(gather_v7_node->get_friendly_name());
        ngraph::copy_runtime_info(gather_v7_node, gather_v1);
        ngraph::replace_node(gather_v7_node, gather_v1);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(gather_v7, matcher_name);
    register_matcher(m, callback);
}
