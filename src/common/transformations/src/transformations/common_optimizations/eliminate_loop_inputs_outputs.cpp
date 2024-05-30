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
#include "openvino/op/tensor_iterator.hpp"
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

    auto subgraph_label = wrap_type<ov::op::v5::Loop, ov::op::v0::TensorIterator>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        using namespace ov::op::util;
        using InvariantD = MultiSubGraphOp::InvariantInputDescription;
        using SlicedD = MultiSubGraphOp::SliceInputDescription;
        using MergedD = MultiSubGraphOp::MergedInputDescription;
        using OutputD = MultiSubGraphOp::BodyOutputDescription;
        using ConcatD = MultiSubGraphOp::ConcatOutputDescription;

        // deleting parameter/results are inserted to the beginning of deque, adding Descriptions to the end
        // deque: delete_param, delete_param, add_input_desc, add_inputs_desc ...
        // delete_param updates Parameter index in the vector, add_input_desc will use the updated indexes.
        std::deque<std::function<void()>> delete_inputs;
        std::deque<std::function<void()>> delete_outputs;
        std::deque<std::function<void()>> process_inputs;
        std::deque<std::function<void()>> process_outputs;

        const auto& pattern_to_output = m.get_pattern_value_map();

        auto subgraph =
            as_type_ptr<ov::op::util::SubGraphOp>(pattern_to_output.at(subgraph_label).get_node_shared_ptr());
        if (!subgraph) {
            return false;
        }
        const auto& body_model = subgraph->get_function();
        const auto& body_params = body_model->get_parameters();
        const auto& body_results = body_model->get_results();

        std::shared_ptr<SubGraphOp> new_node;
        const auto& subgraph_in_values = subgraph->input_values();
        int64_t body_condition_output_idx = -1;
        if (auto loop = as_type_ptr<ov::op::v5::Loop>(subgraph)) {
            const auto& trip_count = subgraph_in_values[0];
            const auto& exec_cond = subgraph_in_values[1];

            auto new_loop = make_shared<ov::op::v5::Loop>(trip_count, exec_cond);
            new_loop->set_special_body_ports(loop->get_special_body_ports());
            new_node = new_loop;
            body_condition_output_idx = loop->get_special_body_ports().body_condition_output_idx;
        } else {
            new_node = make_shared<ov::op::v0::TensorIterator>();
        }
        new_node->set_function(body_model);

        for (const auto& input_description : subgraph->get_input_descriptions()) {
            const auto& body_param = body_params[input_description->m_body_parameter_index];
            const auto& init_value = subgraph_in_values[input_description->m_input_index];

            if (const auto merged_input_desc = as_type_ptr<MergedD>(input_description)) {
                const auto& body_res = body_results[merged_input_desc->m_body_value_index];
                auto param = get_parent_param(body_res);
                if (param && param->get_output_target_inputs(0).size() == 1) {
                    delete_inputs.emplace_front([=, &body_model]() {
                        body_model->remove_parameter(param);
                    });
                } else if (param) {
                    process_inputs.emplace_back([=, &new_node]() {
                        new_node->set_invariant_input(body_param, init_value);
                    });
                } else {
                    process_inputs.emplace_back([=, &new_node]() {
                        new_node->set_merged_input(body_param, init_value, body_res);
                    });
                }
            } else if (const auto invariant_input_desc = as_type_ptr<InvariantD>(input_description)) {
                process_inputs.emplace_back([=, &new_node]() {
                    new_node->set_invariant_input(body_param, init_value);
                });
            } else if (const auto sliced_input_desc = as_type_ptr<SlicedD>(input_description)) {
                process_inputs.emplace_back([=, &new_node]() {
                    new_node->set_sliced_input(body_param,
                                               init_value,
                                               sliced_input_desc->m_start,
                                               sliced_input_desc->m_stride,
                                               sliced_input_desc->m_part_size,
                                               sliced_input_desc->m_end,
                                               sliced_input_desc->m_axis);
                });
            } else {
                // unknown input description type
                // this could only happen if new input description type was added after this transformation written
                return false;
            }
        }
        std::unordered_map<size_t, ov::Output<ov::Node>> loop_inputs;  // body_param->instance_id -> loop_input
        for (const auto& input_description : subgraph->get_input_descriptions()) {
            loop_inputs.emplace(body_params[input_description->m_body_parameter_index]->get_instance_id(),
                                subgraph_in_values[input_description->m_input_index]);
        }
        std::unordered_map<uint64_t, Output<Node>> idx_to_new_output;
        for (const auto& output_description : subgraph->get_output_descriptions()) {
            const auto& out_idx = output_description->m_output_index;
            const auto& body_result = body_results[output_description->m_body_value_index];

            if (output_description->m_body_value_index == body_condition_output_idx) {
                continue;
            }

            const auto& body_param = get_parent_param(body_result);
            if (body_param && output_description->m_body_value_index != body_condition_output_idx) {
                auto new_output = loop_inputs[body_param->get_instance_id()];
                delete_outputs.emplace_front([=, &body_model, &idx_to_new_output]() {
                    body_model->remove_result(body_result);
                    idx_to_new_output[out_idx] = new_output;
                });
            } else if (const auto body_output_desc = as_type_ptr<OutputD>(output_description)) {
                process_outputs.emplace_back([=, &idx_to_new_output, &body_output_desc]() {
                    idx_to_new_output[out_idx] = new_node->get_iter_value(body_result, body_output_desc->m_iteration);
                });
            } else if (const auto concat_output_desc = as_type_ptr<ConcatD>(output_description)) {
                process_outputs.emplace_back([=, &idx_to_new_output, &concat_output_desc]() {
                    idx_to_new_output[out_idx] = new_node->get_concatenated_slices(body_result,
                                                                                   concat_output_desc->m_start,
                                                                                   concat_output_desc->m_stride,
                                                                                   concat_output_desc->m_part_size,
                                                                                   concat_output_desc->m_end,
                                                                                   concat_output_desc->m_axis);
                });
            } else {
                // unknown output description type
                // this could only happen if new output description type was added after this transformation written
                return false;
            }
        }
        if (auto loop = as_type_ptr<ov::op::v5::Loop>(new_node)) {
            auto special_body_ports = loop->get_special_body_ports();
            special_body_ports.body_condition_output_idx -= static_cast<int64_t>(delete_outputs.size());
            loop->set_special_body_ports(special_body_ports);
        }
        for_each(delete_inputs.begin(), delete_inputs.end(), [](function<void()>& f) {
            f();
        });
        for_each(delete_outputs.begin(), delete_outputs.end(), [](function<void()>& f) {
            f();
        });
        for_each(process_inputs.begin(), process_inputs.end(), [](function<void()>& f) {
            f();
        });
        for_each(process_outputs.begin(), process_outputs.end(), [](function<void()>& f) {
            f();
        });

        auto loop_outputs = subgraph->outputs();
        for (size_t i = 0; i < loop_outputs.size(); ++i) {
            loop_outputs[i].replace(idx_to_new_output[i]);
        }
        ov::copy_runtime_info(subgraph, new_node);
        new_node->set_friendly_name(subgraph->get_friendly_name());
        return true;
    };

    auto m = make_shared<Matcher>(subgraph_label, matcher_name);
    this->register_matcher(m, callback);
}
