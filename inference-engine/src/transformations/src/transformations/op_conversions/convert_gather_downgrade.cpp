// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_downgrade.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(pass::ConvertGather7ToGather1, "ConvertGather7ToGather1", 0);
NGRAPH_RTTI_DEFINITION(pass::ConvertGather8ToGather7, "ConvertGather8ToGather7", 0);


template<class SrcOpClass, class DstOpClass>
bool downgrade_gather(ngraph::pattern::Matcher& m,
                       const std::function<std::shared_ptr<DstOpClass>(const std::shared_ptr<SrcOpClass>&)> &dst_node_builder) {
    auto src_node = std::dynamic_pointer_cast<SrcOpClass>(m.get_match_root());
    if (!src_node)
        return false;

    auto dst_node = std::dynamic_pointer_cast<DstOpClass>(dst_node_builder(src_node));
    if (!dst_node)
        return false;

    dst_node->set_friendly_name(src_node->get_friendly_name());
    ngraph::copy_runtime_info(src_node, dst_node);
    ngraph::replace_node(src_node, dst_node);
    return true;
}

pass::ConvertGather7ToGather1::ConvertGather7ToGather1() {
    MATCHER_SCOPE(ConvertGather7ToGather1);

    auto gather_v7_pattern = pattern::wrap_type<opset7::Gather>();
    auto gather_v1_builder = [=] (const shared_ptr<opset7::Gather>& gather_v7_node) {
        shared_ptr<opset1::Gather> gather_v1 = nullptr;

        if (gather_v7_node->get_batch_dims() == 0) {
            gather_v1 = make_shared<opset1::Gather>(gather_v7_node->input_value(0),
                                                    gather_v7_node->input_value(1),
                                                    gather_v7_node->input_value(2));
        }
        return gather_v1;
    };

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        return downgrade_gather<opset7::Gather, opset1::Gather>(m, gather_v1_builder);
    };

    auto m = make_shared<pattern::Matcher>(gather_v7_pattern, matcher_name);
    register_matcher(m, callback);
}

pass::ConvertGather8ToGather7::ConvertGather8ToGather7() {
    MATCHER_SCOPE(ConvertGather8ToGather7);

    auto gather_v8_pattern = pattern::wrap_type<opset8::Gather>();
    auto gather_v7_builder = [=] (const shared_ptr<opset8::Gather>& gather_v8_node) {
        return make_shared<opset7::Gather>(gather_v8_node->input_value(0),
                                           gather_v8_node->input_value(1),
                                           gather_v8_node->input_value(2),
                                           gather_v8_node->get_batch_dims());
    };

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        return downgrade_gather<opset8::Gather, opset7::Gather>(m, gather_v7_builder);
    };

    auto m = make_shared<pattern::Matcher>(gather_v8_pattern, matcher_name);
    register_matcher(m, callback);
}
