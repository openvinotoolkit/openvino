// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_downgrade.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(pass::ConvertGather7ToGather1, "ConvertGather7ToGather1", 0);
NGRAPH_RTTI_DEFINITION(pass::ConvertGather8ToGather7, "ConvertGather8ToGather7", 0);


shared_ptr<opset1::Gather> build_gather_v1(const shared_ptr<opset7::Gather>& gather_v7_node) {
    shared_ptr<opset1::Gather> gather_v1 = nullptr;

    if (gather_v7_node->get_batch_dims() == 0) {
        gather_v1 = make_shared<opset1::Gather>(gather_v7_node->input_value(0),
                                                gather_v7_node->input_value(1),
                                                gather_v7_node->input_value(2));
    }
    return gather_v1;
}

shared_ptr<opset7::Gather> build_gather_v7(const shared_ptr<opset8::Gather>& gather_v8_node) {
    return make_shared<opset7::Gather>(gather_v8_node->input_value(0),
                                       gather_v8_node->input_value(1),
                                       gather_v8_node->input_value(2),
                                       gather_v8_node->get_batch_dims());
}

pass::ConvertGather7ToGather1::ConvertGather7ToGather1() {
    MATCHER_SCOPE(ConvertGather7ToGather1);

    auto gather_v7_pattern = pattern::wrap_type<opset7::Gather>();
    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        return op::util::replace_operation<opset7::Gather, opset1::Gather>(
                m, [](const shared_ptr<opset7::Gather> &gather_v7_node) { return build_gather_v1(gather_v7_node); });
    };

    auto m = make_shared<pattern::Matcher>(gather_v7_pattern, matcher_name);
    register_matcher(m, callback);
}

pass::ConvertGather8ToGather7::ConvertGather8ToGather7() {
    MATCHER_SCOPE(ConvertGather8ToGather7);

    auto gather_v8_pattern = pattern::wrap_type<opset8::Gather>();
    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        return op::util::replace_operation<opset8::Gather, opset7::Gather>(
                m, [](const shared_ptr<opset8::Gather> &gather_v8_node) { return build_gather_v7(gather_v8_node); });
    };

    auto m = make_shared<pattern::Matcher>(gather_v8_pattern, matcher_name);
    register_matcher(m, callback);
}
