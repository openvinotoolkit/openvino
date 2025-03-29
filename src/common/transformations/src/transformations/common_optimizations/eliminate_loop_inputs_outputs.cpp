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
using namespace ov;
using namespace ov::op;
using namespace ov::op::util;
using namespace ov::element;
using namespace ov::pass::pattern;

using InvariantD = MultiSubGraphOp::InvariantInputDescription;
using SlicedD = MultiSubGraphOp::SliceInputDescription;
using MergedD = MultiSubGraphOp::MergedInputDescription;
using OutputD = MultiSubGraphOp::BodyOutputDescription;
using ConcatD = MultiSubGraphOp::ConcatOutputDescription;

using ResultPtr = std::shared_ptr<v0::Result>;
using OutputDescPtr = MultiSubGraphOp::OutputDescription::Ptr;
using OutputDescMap = std::unordered_map<ResultPtr, OutputDescPtr>;
using InputDescPtr = MultiSubGraphOp::InputDescription::Ptr;
using BodyResultIdxMap = std::unordered_map<ResultPtr, uint64_t>;

namespace {
std::unordered_set<uint64_t> remove_results(const std::shared_ptr<v0::Parameter>& param,
                                            const InputDescPtr& input_desc,
                                            OutputDescMap& output_desc_map,
                                            const BodyResultIdxMap& result_map,
                                            int64_t current_iteration_input_idx,
                                            int64_t body_condition_output_idx) {
    if (current_iteration_input_idx == static_cast<int64_t>(input_desc->m_body_parameter_index)) {
        // skip Parameter node storing the current iteration
        return {};
    }

    // remove sub-graphs when Parameter is going to only Result nodes
    // but Parameter node must be marked as Invariant or Merged Parameter
    // and Result node must be unmarked (no OutputDescription) or marked as BodyOutputDescription
    // all other cases do not guarantee to leave a graph valid
    if (!ov::as_type_ptr<InvariantD>(input_desc) && !ov::as_type_ptr<MergedD>(input_desc)) {
        return {};
    }

    std::unordered_set<uint64_t> removed_result_inds;
    for (const auto& target_input : param->get_output_target_inputs(0)) {
        const auto& consumer = target_input.get_node()->shared_from_this();
        const auto& result = ov::as_type_ptr<v0::Result>(consumer);
        if (!result) {
            continue;
        }
        if (static_cast<int64_t>(result_map.at(result)) == body_condition_output_idx) {
            // body condition output and related Parameter node must not be removed
            continue;
        }

        const auto& output_desc = output_desc_map[result];
        if ((output_desc && ov::as_type_ptr<OutputD>(output_desc)) || !output_desc) {
            removed_result_inds.insert(result_map.at(result));
        } else {
            // unknown or unsupported case is met
            continue;
        }
    }

    if (const auto merged_input_desc = ov::as_type_ptr<MergedD>(input_desc)) {
        if (removed_result_inds.count(merged_input_desc->m_body_value_index) > 0) {
            return removed_result_inds;
        }
        return {};
    }

    return removed_result_inds;
}
}  // namespace

ov::pass::EliminateLoopInputsOutputs::EliminateLoopInputsOutputs() {
    MATCHER_SCOPE(EliminateLoopInputsOutputs);

    auto subgraph_label = wrap_type<v5::Loop, v0::TensorIterator>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        // delete useless Parameter and Result nodes in isolated graph (Parameter->Result)
        // after that, set indices of sliced and invariant inputs
        // and outputs in the resulted SubGraph operation
        std::deque<std::function<void()>> delete_inputs_outputs;
        std::deque<std::function<void()>> process_inputs_outputs;

        const auto& pattern_to_output = m.get_pattern_value_map();

        auto subgraph = as_type_ptr<SubGraphOp>(pattern_to_output.at(subgraph_label).get_node_shared_ptr());
        if (!subgraph) {
            return false;
        }
        const auto& body_model = subgraph->get_function();
        const auto& body_params = body_model->get_parameters();
        const auto& body_results = body_model->get_results();

        std::shared_ptr<SubGraphOp> new_node;
        const auto& subgraph_in_values = subgraph->input_values();
        int64_t body_condition_output_idx = -1;
        int64_t current_iteration_input_idx = -1;
        if (auto loop = as_type_ptr<v5::Loop>(subgraph)) {
            const auto& trip_count = subgraph_in_values[0];
            const auto& exec_cond = subgraph_in_values[1];

            auto new_loop = make_shared<v5::Loop>(trip_count, exec_cond);
            new_loop->set_special_body_ports(loop->get_special_body_ports());
            new_node = new_loop;
            // condition Result node index may be shifted due to removing
            // useless Parameter->Result subgraphs in the body
            // save initial body_condition_output_idx before removing
            body_condition_output_idx = loop->get_special_body_ports().body_condition_output_idx;
            current_iteration_input_idx = loop->get_special_body_ports().current_iteration_input_idx;
        } else {
            new_node = make_shared<v0::TensorIterator>();
        }
        new_node->set_function(body_model);

        // walk through OutputDescription vector and store marks for body Result nodes
        OutputDescMap output_desc_map;
        for (const auto& output_description : subgraph->get_output_descriptions()) {
            const auto& body_result = body_results[output_description->m_body_value_index];
            output_desc_map[body_result] = output_description;
        }
        BodyResultIdxMap result_map;
        for (uint64_t idx = 0; idx < body_results.size(); ++idx) {
            result_map[body_results[idx]] = idx;
        }
        std::unordered_map<size_t, Output<Node>> loop_inputs;  // body_param->instance_id -> loop_input
        for (const auto& input_description : subgraph->get_input_descriptions()) {
            loop_inputs.emplace(body_params[input_description->m_body_parameter_index]->get_instance_id(),
                                subgraph_in_values[input_description->m_input_index]);
        }

        // collect Parameter and Result nodes to be removed
        // some isolated sub-graphs (Parameter->Result) are removed
        std::unordered_set<uint64_t> remove_result_inds;
        // calculate a number of removed Result nodes standing before body condition Result
        // it needs to update body_condition_output_idx
        int64_t delta_body_condition_output_idx = 0;
        int64_t delta_current_iteration_input_idx = 0;
        // a map of new producers by each Loop/TI output index
        std::unordered_map<uint64_t, Output<Node>> idx_to_new_output;

        for (const auto& input_description : subgraph->get_input_descriptions()) {
            auto body_param = body_params[input_description->m_body_parameter_index];
            const auto& init_value = subgraph_in_values[input_description->m_input_index];

            auto cur_remove_result_inds = remove_results(body_param,
                                                         input_description,
                                                         output_desc_map,
                                                         result_map,
                                                         current_iteration_input_idx,
                                                         body_condition_output_idx);
            if (cur_remove_result_inds.size() > 0) {
                for (const auto& result_idx : cur_remove_result_inds) {
                    auto body_result = body_results[result_idx];
                    if (const auto& output_desc = output_desc_map[body_result]) {
                        auto out_idx = output_desc->m_output_index;
                        auto new_output = loop_inputs[body_param->get_instance_id()];
                        delete_inputs_outputs.emplace_front([=, &body_model, &idx_to_new_output]() {
                            body_model->remove_result(body_result);
                            idx_to_new_output[out_idx] = new_output;
                        });
                    } else {
                        delete_inputs_outputs.emplace_front([=, &body_model]() {
                            body_model->remove_result(body_result);
                        });
                    }

                    if (static_cast<int64_t>(result_idx) < body_condition_output_idx) {
                        ++delta_body_condition_output_idx;
                    }
                    remove_result_inds.insert(result_idx);
                }
            }

            if (body_param->get_output_target_inputs(0).size() == cur_remove_result_inds.size()) {
                // remove Parameter node since all consumers (Result nodes) will be also removed
                delete_inputs_outputs.emplace_front([=, &body_model]() {
                    body_model->remove_parameter(body_param);
                });
                if (static_cast<int64_t>(input_description->m_body_parameter_index) < current_iteration_input_idx) {
                    ++delta_current_iteration_input_idx;
                }
                // nothing to mark with input description
                continue;
            }

            // move input description for unremoved node
            if (const auto merged_input_desc = as_type_ptr<MergedD>(input_description)) {
                if (cur_remove_result_inds.size() > 0) {
                    // at least one Result node is going to be removed
                    // no back edge after removing so use invariant input
                    process_inputs_outputs.emplace_back([=, &new_node]() {
                        new_node->set_invariant_input(body_param, init_value);
                    });
                } else {
                    const auto& body_res = body_results[merged_input_desc->m_body_value_index];
                    process_inputs_outputs.emplace_back([=, &new_node]() {
                        new_node->set_merged_input(body_param, init_value, body_res);
                    });
                }
            } else if (const auto invariant_input_desc = as_type_ptr<InvariantD>(input_description)) {
                process_inputs_outputs.emplace_back([=, &new_node]() {
                    new_node->set_invariant_input(body_param, init_value);
                });
            } else if (const auto sliced_input_desc = as_type_ptr<SlicedD>(input_description)) {
                process_inputs_outputs.emplace_back([=, &new_node]() {
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

        if (delete_inputs_outputs.size() == 0) {
            // nothing is to remove
            return false;
        }

        for (const auto& output_description : subgraph->get_output_descriptions()) {
            const auto& out_idx = output_description->m_output_index;
            const auto& body_result = body_results[output_description->m_body_value_index];

            if (remove_result_inds.count(output_description->m_body_value_index) > 0) {
                // the corresponsing result will be removed so it needs to insert skip connection
                continue;
            }

            if (const auto body_output_desc = as_type_ptr<OutputD>(output_description)) {
                process_inputs_outputs.emplace_back([=, &idx_to_new_output]() {
                    idx_to_new_output[out_idx] = new_node->get_iter_value(body_result, body_output_desc->m_iteration);
                });
            } else if (const auto concat_output_desc = as_type_ptr<ConcatD>(output_description)) {
                process_inputs_outputs.emplace_back([=, &idx_to_new_output]() {
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

        if (auto loop = as_type_ptr<v5::Loop>(new_node)) {
            // update of body condition index is required due to body graph clean-up
            auto special_body_ports = loop->get_special_body_ports();
            if (special_body_ports.body_condition_output_idx >= 0) {
                special_body_ports.body_condition_output_idx -= delta_body_condition_output_idx;
            }

            if (special_body_ports.current_iteration_input_idx >= 0) {
                special_body_ports.current_iteration_input_idx -= delta_current_iteration_input_idx;
            }
            loop->set_special_body_ports(special_body_ports);
        }

        for_each(delete_inputs_outputs.begin(), delete_inputs_outputs.end(), [](function<void()>& f) {
            f();
        });
        for_each(process_inputs_outputs.begin(), process_inputs_outputs.end(), [](function<void()>& f) {
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
