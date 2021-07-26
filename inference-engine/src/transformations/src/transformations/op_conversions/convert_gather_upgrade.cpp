// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_upgrade.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(pass::ConvertGather1ToGather7, "ConvertGather1ToGather7", 0);
NGRAPH_RTTI_DEFINITION(pass::ConvertGather7ToGather8, "ConvertGather7ToGather8", 0);

shared_ptr<opset8::Gather> build_gather_v8(const shared_ptr<opset7::Gather>& gather_v7_node) {
        return make_shared<opset8::Gather>(gather_v7_node->input_value(0),
                                           gather_v7_node->input_value(1),
                                           gather_v7_node->input_value(2),
                                           gather_v7_node->get_batch_dims());
}

shared_ptr<opset7::Gather> build_gather_v7(const shared_ptr<opset1::Gather>& gather_v1_node) {
    return make_shared<opset7::Gather>(gather_v1_node->input_value(0),
                                       gather_v1_node->input_value(1),
                                       gather_v1_node->input_value(2),
                                       0);
}

pass::ConvertGather1ToGather7::ConvertGather1ToGather7() {
    MATCHER_SCOPE(ConvertGather1ToGather7);

    auto gather_v1 = pattern::wrap_type<opset1::Gather>();
    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        return op::util::replace_operation<opset1::Gather, opset7::Gather>(
                m, [](const shared_ptr<opset1::Gather> &gather_v1_node) { return build_gather_v7(gather_v1_node); });
    };

    auto m = make_shared<pattern::Matcher>(gather_v1, matcher_name);
    register_matcher(m, callback);
}

pass::ConvertGather7ToGather8::ConvertGather7ToGather8() {
    MATCHER_SCOPE(ConvertGather7ToGather8);

    auto gather_v7 = pattern::wrap_type<opset7::Gather>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        return op::util::replace_operation<opset7::Gather, opset8::Gather>(
                m, [](const shared_ptr<opset7::Gather> &gather_v7_node) { return build_gather_v8(gather_v7_node); });
    };

    auto m = make_shared<pattern::Matcher>(gather_v7, matcher_name);
    register_matcher(m, callback);
}
