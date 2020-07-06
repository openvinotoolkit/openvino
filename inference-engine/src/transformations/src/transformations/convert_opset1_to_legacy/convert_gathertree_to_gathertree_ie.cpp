// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_gathertree_to_gathertree_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

ngraph::pass::ConvertGatherTreeToGatherTreeIEMatcher::ConvertGatherTreeToGatherTreeIEMatcher() {
    auto input0 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1});
    auto input1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1});
    auto input2 = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto input3 = std::make_shared<pattern::op::Label>(element::i64, Shape{});
    auto gt = std::make_shared<ngraph::opset1::GatherTree>(input0, input1, input2, input3);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto gt = std::dynamic_pointer_cast<ngraph::opset1::GatherTree> (m.get_match_root());
        if (!gt) {
            return false;
        }
        auto reshape = std::make_shared<opset1::Reshape>(gt->input_value(3),
                                                         opset1::Constant::create<int64_t>(element::i64, Shape{1}, {1}),
                                                         true);
        auto gt_ie = std::make_shared<ngraph::op::GatherTreeIE>(gt->input_value(0), gt->input_value(1), gt->input_value(2), reshape);

        gt_ie->set_friendly_name(gt->get_friendly_name());
        ngraph::copy_runtime_info(gt, {reshape, gt_ie});
        ngraph::replace_node(gt, gt_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gt, "ConvertGatherTreeToGatherTreeIE");
    this->register_matcher(m, callback);
}