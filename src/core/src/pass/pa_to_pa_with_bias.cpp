// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pa_to_pa_with_bias.hpp"
#include "openvino/op/paged_attention_with_qq_bias.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/core/graph_util.hpp"
#include <openvino/core/rt_info.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/graph_rewrite.hpp>
using namespace ov::op;

bool ov::pass::PAToPAWithQQBias::run_on_model(const std::shared_ptr<ov::Model>& model) {
    auto qq_bias_input = std::make_shared<ov::op::v0::Parameter>(
            ov::element::i8, PartialShape{-1, -1, -1}); // shape will be inferred later
    qq_bias_input->set_friendly_name("qq_bias");
    bool found = false;
    for (const auto& node : model->get_ordered_ops()) {
        auto pa_op = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node);
        if (!pa_op) continue;
        found = true;
        OutputVector inputs;
        for (size_t i = 0; i < pa_op->get_input_size(); i++) {
            inputs.push_back(pa_op->input_value(i));
        }
        // add qq_bias input
        inputs.push_back(qq_bias_input);
        // create new PagedAttentionExtension with qq_bias input
        auto new_pa_ext_node = std::make_shared<ov::op::paged_attention_with_qq_bias>(inputs);
        new_pa_ext_node->set_friendly_name(pa_op->get_friendly_name());
        ov::copy_runtime_info(pa_op, new_pa_ext_node);
        ov::replace_node(pa_op, new_pa_ext_node);
    }
    if(found)
        model->add_parameters({qq_bias_input});
    return true;
}
