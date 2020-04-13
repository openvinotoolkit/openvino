// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_topk_to_topk_ie.hpp"

#include <memory>
#include <vector>
#include <string>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/topk_ie.hpp>

void ngraph::pass::ConvertTopKToTopKIE::convert_topk_to_topk_ie() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto k = std::make_shared<pattern::op::Label>(element::i64, Shape{});
    auto topk = std::make_shared<ngraph::opset1::TopK>(input_0, k, 0, "min", "none");

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
        auto topk = std::dynamic_pointer_cast<ngraph::opset1::TopK>(m.get_match_root());
        if (!topk) {
            return false;
        }
        if (topk->input(1).get_shape().size() == 1) {
            return false;
        }
        auto unsqueezed_k = std::make_shared<ngraph::opset1::Unsqueeze>(topk->input(1).get_source_output().get_node_shared_ptr(),
                                                                        opset1::Constant::create(element::i64, Shape{1}, {0}));

        std::string mode;
        switch (topk->get_mode()) {
            case ngraph::opset1::TopK::Mode::MAX:
                mode = "max";
                break;
            case ngraph::opset1::TopK::Mode::MIN:
                mode = "min";
                break;
            default:
                return false;
        }
        std::string sort_type;
        switch (topk->get_sort_type()) {
            case ngraph::opset1::TopK::SortType::NONE:
                sort_type = "none";
                break;
            case ngraph::opset1::TopK::SortType::SORT_INDICES:
                sort_type = "index";
                break;
            case ngraph::opset1::TopK::SortType::SORT_VALUES:
                sort_type = "value";
                break;
            default:
                return false;
        }

        auto new_topk = std::make_shared<ngraph::op::TopKIE>(topk->input(0).get_source_output(), unsqueezed_k, topk->get_axis(), mode,
                                                             sort_type, topk->output(0).get_shape());
        new_topk->set_friendly_name(topk->get_friendly_name());
        ngraph::replace_node(m.get_match_root(), new_topk);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(topk, "ConvertTopKToTopKIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}