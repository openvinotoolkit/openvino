// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset3_to_opset2/convert_topk3.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertTopK3::convert_topk3() {
    auto topk = std::make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<opset3::TopK>());

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto topk = std::dynamic_pointer_cast<ngraph::opset3::TopK> (m.get_match_root());
        if (!topk) {
            return false;
        }
        Output<Node> last0;
        Output<Node> last1;
        ngraph::NodeVector new_ops;

        auto new_topk = std::make_shared<ngraph::opset2::TopK>(topk->input_value(0), topk->input_value(1),
                topk->get_axis(), topk->get_mode(), topk->get_sort_type(), element::i32);
        new_ops.push_back(new_topk);
        // if the output is the i32 then it matches behavior of the v1::TopK otherwise need to insert Convert
        if (topk->get_index_element_type() == element::i32) {
            last0 = new_topk->output(0);
            last1 = new_topk->output(1);
            new_topk->set_friendly_name(topk->get_friendly_name());
        } else {
            // create fake convert for 0 output, it is a workaround in purpose of correct output names preserving
            last0 = std::make_shared<ngraph::opset2::Convert>(new_topk->output(0), topk->get_output_element_type(0));
            last1 = std::make_shared<ngraph::opset2::Convert>(new_topk->output(1), topk->get_index_element_type());
            new_ops.push_back(last0.get_node_shared_ptr());
            new_ops.push_back(last1.get_node_shared_ptr());

            // workaround for naming two outputs of TopK
            last0.get_node_shared_ptr()->set_friendly_name(topk->get_friendly_name() + ".0");
            last1.get_node_shared_ptr()->set_friendly_name(topk->get_friendly_name() + ".1");
        }

        ngraph::copy_runtime_info(topk, new_ops);
        topk->output(0).replace(last0);
        topk->output(1).replace(last1);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(topk, "ConvertTopK3");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
