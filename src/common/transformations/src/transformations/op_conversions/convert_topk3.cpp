// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_topk3.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertTopK3::ConvertTopK3() {
    MATCHER_SCOPE(ConvertTopK3);
    auto topk = pattern::wrap_type<ov::op::v3::TopK>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto topk = ov::as_type_ptr<ov::op::v3::TopK>(m.get_match_root());
        if (!topk) {
            return false;
        }
        Output<Node> last0;
        Output<Node> last1;
        ov::NodeVector new_ops;

        auto new_topk = std::make_shared<ov::op::v1::TopK>(topk->input_value(0),
                                                           topk->input_value(1),
                                                           topk->get_axis(),
                                                           topk->get_mode(),
                                                           topk->get_sort_type(),
                                                           element::i32);
        new_ops.push_back(new_topk);
        // if the output is the i32 or output #1 has no consumers
        // then it matches behavior of the v1::TopK otherwise need to insert Convert
        if (topk->get_index_element_type() == element::i32 || topk->get_output_target_inputs(1).size() == 0) {
            last0 = new_topk->output(0);
            last1 = new_topk->output(1);
            new_topk->set_friendly_name(topk->get_friendly_name());
        } else if (topk->get_output_target_inputs(0).size() == 0) {
            last0 = topk->output(0);
            last1 = std::make_shared<ov::op::v0::Convert>(new_topk->output(1), topk->get_index_element_type());
            new_ops.push_back(last1.get_node_shared_ptr());

            // workaround for naming two outputs of TopK
            last1.get_node_shared_ptr()->set_friendly_name(topk->get_friendly_name() + ".1");
        } else {
            // create fake convert for 0 output, it is a workaround in purpose of correct output names preserving
            last0 = std::make_shared<ov::op::v0::Convert>(new_topk->output(0), topk->get_output_element_type(0));
            last1 = std::make_shared<ov::op::v0::Convert>(new_topk->output(1), topk->get_index_element_type());
            new_ops.push_back(last0.get_node_shared_ptr());
            new_ops.push_back(last1.get_node_shared_ptr());

            // workaround for naming two outputs of TopK
            last0.get_node_shared_ptr()->set_friendly_name(topk->get_friendly_name() + ".0");
            last1.get_node_shared_ptr()->set_friendly_name(topk->get_friendly_name() + ".1");
        }

        ov::copy_runtime_info(topk, new_ops);
        topk->output(0).replace(last0);
        topk->output(1).replace(last1);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(topk, matcher_name);
    register_matcher(m, callback);
}
