// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_topk_to_topk_ie.hpp"

#include <memory>
#include <vector>
#include <string>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/topk_ie.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertTopKToTopKIE::convert_topk_to_topk_ie() {
    auto topk = std::make_shared<pattern::op::Label>(element::f32, Shape{1}, pattern::has_class<opset1::TopK>());

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
        auto topk = std::dynamic_pointer_cast<opset1::TopK>(m.get_match_root());
        if (!topk || topk->input(1).get_partial_shape().rank().is_dynamic()) {
            return false;
        }
        if (topk->input(1).get_partial_shape().rank().get_length() == 1) {
            return false;
        }
        auto unsqueezed_k = std::make_shared<opset1::Unsqueeze>(topk->input(1).get_source_output().get_node_shared_ptr(),
                                                                        opset1::Constant::create(element::i64, Shape{1}, {0}));

        auto new_topk = std::make_shared<ngraph::op::TopKIE>(topk->input_value(0), unsqueezed_k, topk->get_axis(), topk->get_mode(),
                                                             topk->get_sort_type());
        new_topk->set_friendly_name(topk->get_friendly_name());
        ngraph::copy_runtime_info(topk, {unsqueezed_k, new_topk});
        ngraph::replace_node(topk, new_topk);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(topk, "ConvertTopKToTopKIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}