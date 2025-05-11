// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/pass/transform_while.hpp"

#include "default_opset.hpp"
#include "internal/op/while.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/paddle/exception.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/fold_subgraph_empty_inputs.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace ov::frontend::paddle::op::default_opset;

// Transform Paddle "while" to OpenVINO Loop op.
// "set_merged_input" is used for possible cases like TensorArray
// when it is both the input and the output to the loop.
// The reason why not using concat the output (i.e. "get_concatenated_slices") here is that,
// it would complicate processing TensorArray.
// TensorArray could be in loop body, but it could not always append something;
// TensorArray could be a non-empty input of the loop body, which needs extra concat.
// What's more, we have to tell which output is of TensorArray type to concate.
ov::frontend::paddle::pass::TransformWhile::TransformWhile(std::vector<std::shared_ptr<Model>> functions) {
    const auto while_label = pattern::wrap_type<ov::op::internal::While>();

    matcher_pass_callback callback = [functions](pattern::Matcher& m) -> bool {
        const auto& while_node = ov::as_type_ptr<ov::op::internal::While>(m.get_match_root());
        if (!while_node)
            return false;
        const auto& inputs = while_node->input_values();
        const auto trip_count = Constant::create(element::i64, {1}, {-1});
        const auto& cond = inputs.back();
        const auto cond_name = cond.get_node_shared_ptr()->get_friendly_name();
        auto loop = std::make_shared<Loop>(trip_count, cond);
        ov::pass::disable_fold_subgraph_empty_inputs(loop);
        const auto block_idx = while_node->get_subblock_index();
        const auto sub_model = functions[block_idx];
        loop->set_function(sub_model);

        const auto& parameters = sub_model->get_parameters();
        const auto submodel_outputs = sub_model->outputs();
        const auto is_exist = [&submodel_outputs](const std::string& name) {
            for (const auto& out : submodel_outputs) {
                if (out.get_any_name() == name)
                    return true;
            }
            return false;
        };
        for (size_t i = 0; i < parameters.size(); i++) {
            const auto names = inputs[i].get_names();
            std::string param_name;
            if (!names.empty()) {
                param_name = *names.begin();
            }
            if (!param_name.empty() && is_exist(param_name)) {
                auto out_node = sub_model->output(param_name);
                loop->set_merged_input(parameters[i], inputs[i], out_node);
            } else {
                loop->set_invariant_input(parameters[i], inputs[i]);
            }
        }
        int64_t idx = -1;
        for (size_t i = 0; i < sub_model->get_results().size(); i++) {
            if (sub_model->output(i).get_tensor().get_any_name() == cond_name)
                idx = static_cast<int64_t>(i);
        }
        FRONT_END_GENERAL_CHECK(idx != -1, "could not find condition node in outputs.");

        loop->set_special_body_ports(Loop::SpecialBodyPorts{-1, idx});

        // replace output
        const auto& results = sub_model->get_results();
        for (size_t i = 0; i < results.size(); i++) {
            auto out = loop->get_iter_value(results[i], -1);
            while_node->output(i).replace(out);
        }

        loop->add_node_control_dependents(while_node);
        loop->add_node_control_dependencies(while_node);
        while_node->clear_control_dependents();

        loop->set_friendly_name(while_node->get_friendly_name());
        copy_runtime_info(while_node, loop);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(while_label, "while_loop");
    this->register_matcher(m, callback);
}
