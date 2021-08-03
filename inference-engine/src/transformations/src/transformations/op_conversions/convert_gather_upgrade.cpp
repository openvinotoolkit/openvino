// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_upgrade.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(pass::ConvertGather1ToGather7, "ConvertGather1ToGather7", 0);
NGRAPH_RTTI_DEFINITION(pass::ConvertGather7ToGather8, "ConvertGather7ToGather8", 0);


template<class SrcOpClass, class DstOpClass>
bool upgrade_gather(ngraph::pattern::Matcher& m,
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

pass::ConvertGather1ToGather7::ConvertGather1ToGather7() {
    MATCHER_SCOPE(ConvertGather1ToGather7);

    auto gather_v1_pattern = pattern::wrap_type<opset1::Gather>();
    auto gather_v7_builder = [=](const std::shared_ptr<opset1::Gather>& gather_v1_node){
        return make_shared<opset7::Gather>(gather_v1_node->input_value(0),
                                           gather_v1_node->input_value(1),
                                           gather_v1_node->input_value(2),
                                           0);
    };

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        return upgrade_gather<opset1::Gather, opset7::Gather>(m, gather_v7_builder);
    };

    auto m = make_shared<pattern::Matcher>(gather_v1_pattern, matcher_name);
    register_matcher(m, callback);
}

pass::ConvertGather7ToGather8::ConvertGather7ToGather8() {
    MATCHER_SCOPE(ConvertGather7ToGather8);

    auto gather_v7_pattern = pattern::wrap_type<opset7::Gather>();
    auto gather_v8_builder = [=](const std::shared_ptr<opset7::Gather>& gather_v7_node){
        return make_shared<opset8::Gather>(gather_v7_node->input_value(0),
                                           gather_v7_node->input_value(1),
                                           gather_v7_node->input_value(2),
                                           gather_v7_node->get_batch_dims());
    };

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        return upgrade_gather<opset7::Gather, opset8::Gather>(m, gather_v8_builder);
    };

    auto m = make_shared<pattern::Matcher>(gather_v7_pattern, matcher_name);
    register_matcher(m, callback);
}
