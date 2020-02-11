// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

#include <ngraph/op/gather_tree.hpp>
#include <ngraph_ops/gather_tree_ie.hpp>

namespace ngraph {
namespace pass {

class ConvertGatherTreeToGatherTreeIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertGatherTreeToGatherTreeIE: public ngraph::pass::GraphRewrite {
public:
    ConvertGatherTreeToGatherTreeIE() : GraphRewrite() {
        convert();
    }

private:
    void convert() {
        auto input0 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1});
        auto input1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1});
        auto input2 = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
        auto input3 = std::make_shared<pattern::op::Label>(element::i64, Shape{});
        auto gt = std::make_shared<ngraph::op::v1::GatherTree>(input0, input1, input2, input3);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto gt = std::dynamic_pointer_cast<ngraph::op::v1::GatherTree> (m.get_match_root());
            if (!gt) {
                return false;
            }
            auto reshape = std::make_shared<op::v1::Reshape>(gt->input_value(3),
                                                             op::Constant::create<int64_t>(element::i64, Shape{1}, {1}),
                                                             true);
            auto gt_ie = std::make_shared<ngraph::op::GatherTreeIE>(gt->input_value(0), gt->input_value(1), gt->input_value(2), reshape);

            gt_ie->set_friendly_name(gt->get_friendly_name());
            ngraph::replace_node(gt, gt_ie);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(gt, "ConvertGatherTreeToGatherTreeIE");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
