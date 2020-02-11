// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ngraph/pass/graph_rewrite.hpp>

#include "ngraph/op/topk.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/fused/unsqueeze.hpp"
#include <ngraph_ops/topk_ie.hpp>

namespace ngraph {
namespace pass {

class ConvertTopKToTopKIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertTopKToTopKIE : public ngraph::pass::GraphRewrite {
public:
    ConvertTopKToTopKIE() : GraphRewrite() {
        convert_topk_to_topk_ie();
    }

private:
    void convert_topk_to_topk_ie();
};

void ngraph::pass::ConvertTopKToTopKIE::convert_topk_to_topk_ie() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto k = std::make_shared<pattern::op::Label>(element::i64, Shape{});
    auto topk = std::make_shared<ngraph::op::v1::TopK>(input_0, k, 0, "min", "none");

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
        auto topk = std::dynamic_pointer_cast<ngraph::op::v1::TopK>(m.get_match_root());
        if (!topk) {
            return false;
        }
        if (topk->input(1).get_shape().size() == 1) {
            return false;
        }
        auto unsqueezed_k = std::make_shared<ngraph::op::Unsqueeze>(topk->input(1).get_source_output().get_node_shared_ptr(),
                                                                    op::Constant::create(element::i64, Shape{1}, {0}));

        std::string mode;
        switch (topk->get_mode()) {
        case ngraph::op::v1::TopK::Mode::MAX:
            mode = "max";
            break;
        case ngraph::op::v1::TopK::Mode::MIN:
            mode = "min";
            break;
        default:
            return false;
        }
        std::string sort_type;
        switch (topk->get_sort_type()) {
            case ngraph::op::v1::TopK::SortType::NONE:
                sort_type = "none";
                break;
            case ngraph::op::v1::TopK::SortType::SORT_INDICES:
                sort_type = "index";
                break;
            case ngraph::op::v1::TopK::SortType::SORT_VALUES:
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
