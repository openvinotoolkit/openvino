// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_loop_inputs_outputs.hpp"

#include <unordered_map>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace std;
using namespace ov::element;
using namespace ov::pass::pattern;

namespace {
std::shared_ptr<ov::op::v0::Parameter> get_parent_param(const std::shared_ptr<ov::op::v0::Result>& result) {
    return ov::as_type_ptr<ov::op::v0::Parameter>(result->input_values()[0].get_node_shared_ptr());
}
}  // namespace

ov::pass::EliminateLoopInputsOutputs::EliminateLoopInputsOutputs() {
    MATCHER_SCOPE(EliminateLoopInputsOutputs);

    auto loop_label = wrap_type<ov::op::v5::Loop>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto loop = as_type_ptr<ov::op::v5::Loop>(pattern_to_output.at(loop_label).get_node_shared_ptr());
        auto body_model = loop->get_function();
        const auto body_params = body_model->get_parameters();
        const auto body_results = body_model->get_results();

        const auto loop_input_values = loop->input_values();
        const auto& trip_count = loop_input_values[0];
        const auto& exec_cond = loop_input_values[1];

        auto new_loop = make_shared<ov::op::v5::Loop>(trip_count, exec_cond);
        new_loop->set_function(body_model);
        new_loop->set_special_body_ports(loop->get_special_body_ports());

        for (const auto& input_description : loop->get_input_descriptions()) {
            if (const auto merged_input_desc =
                    as_type_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>(input_description)) {
                if (get_parent_param(body_results[merged_input_desc->m_body_value_index])) {
                    new_loop->set_invariant_input(body_params[merged_input_desc->m_body_parameter_index],
                                                  loop_input_values[merged_input_desc->m_input_index]);
                } else {
                    new_loop->set_merged_input(body_params[merged_input_desc->m_body_parameter_index],
                                               loop_input_values[merged_input_desc->m_input_index],
                                               body_results[merged_input_desc->m_body_value_index]);
                }
            } else if (const auto invariant_input_desc =
                           as_type_ptr<ov::op::util::MultiSubGraphOp::InvariantInputDescription>(input_description)) {
                new_loop->set_invariant_input(body_params[invariant_input_desc->m_body_parameter_index],
                                              loop_input_values[invariant_input_desc->m_input_index]);
            } else if (const auto sliced_input_desc =
                           as_type_ptr<ov::op::util::MultiSubGraphOp::SliceInputDescription>(input_description)) {
                new_loop->set_sliced_input(body_params[sliced_input_desc->m_body_parameter_index],
                                           loop_input_values[sliced_input_desc->m_input_index],
                                           sliced_input_desc->m_start,
                                           sliced_input_desc->m_stride,
                                           sliced_input_desc->m_part_size,
                                           sliced_input_desc->m_end,
                                           sliced_input_desc->m_axis);
            } else {
                /* unknown input description type
                 * this could only happen if new input description type was added after this transformation
                 * written
                 */
                return false;
            }
        }

        std::unordered_map<size_t, ov::Output<ov::Node>> loop_inputs;  // body_param->instance_id -> loop_input
        for (const auto& input_description : loop->get_input_descriptions()) {
            loop_inputs.emplace(body_params[input_description->m_body_parameter_index]->get_instance_id(),
                                loop_input_values[input_description->m_input_index]);
        }

        ov::OutputVector new_loop_outputs;
        int64_t iteration = -1;
        for (const auto& output_description : loop->get_output_descriptions()) {
            iteration = -1;
            if (const auto body_output_desc =
                    as_type_ptr<ov::op::util::MultiSubGraphOp::BodyOutputDescription>(output_description)) {
                iteration = body_output_desc->m_iteration;
            }
            const auto& body_result = body_results[output_description->m_body_value_index];
            if (const auto& body_param = get_parent_param(body_result)) {
                new_loop_outputs.emplace_back(loop_inputs[body_param->get_instance_id()]);
            } else {
                new_loop_outputs.emplace_back(new_loop->get_iter_value(body_result, iteration));
            }
        }

        auto loop_outputs = loop->outputs();
        for (size_t i = 0; i < loop_outputs.size(); ++i) {
            loop_outputs[i].replace(new_loop_outputs[i]);
        }

        ov::copy_runtime_info(loop, new_loop);
        new_loop->set_friendly_name(loop->get_friendly_name());

        return true;
    };

    auto m = make_shared<Matcher>(loop_label, matcher_name);
    this->register_matcher(m, callback);
}
