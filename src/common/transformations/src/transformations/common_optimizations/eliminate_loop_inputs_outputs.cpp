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
/*
 * As we remove some body model parameters and results, we divide obtaining previous loop input/output
 * descriptions and setting new loop descriptions. We remove parameters and results between these two
 * steps. The next classes are served to store data from previous loop and set up input/output data
 * to the new loop.
 */
class SubgraphInput {
public:
    SubgraphInput() = default;
    virtual ~SubgraphInput() = default;
    virtual void add(std::shared_ptr<ov::op::util::SubGraphOp> subgraph) const = 0;
};

class InvariantInput : public SubgraphInput {
public:
    InvariantInput(const std::shared_ptr<ov::op::v0::Parameter>& body_parameter, const ov::Output<ov::Node>& value)
        : m_body_parameter(body_parameter),
          m_value(value) {}
    void add(std::shared_ptr<ov::op::util::SubGraphOp> subgraph) const override {
        subgraph->set_invariant_input(m_body_parameter, m_value);
    }

private:
    const std::shared_ptr<ov::op::v0::Parameter> m_body_parameter;
    const ov::Output<ov::Node> m_value;
};

class MergedInput : public SubgraphInput {
public:
    MergedInput(const std::shared_ptr<ov::op::v0::Parameter>& body_parameter,
                const ov::Output<ov::Node>& initial_value,
                const ov::Output<ov::Node>& successive_value)
        : m_body_parameter(body_parameter),
          m_initial_value(initial_value),
          m_successive_value(successive_value) {}
    void add(std::shared_ptr<ov::op::util::SubGraphOp> subgraph) const override {
        subgraph->set_merged_input(m_body_parameter, m_initial_value, m_successive_value);
    }

private:
    const std::shared_ptr<ov::op::v0::Parameter> m_body_parameter;
    const ov::Output<ov::Node> m_initial_value;
    const ov::Output<ov::Node> m_successive_value;
};

class SlicedInput : public SubgraphInput {
public:
    SlicedInput(const std::shared_ptr<ov::op::v0::Parameter>& body_parameter,
                const ov::Output<ov::Node>& value,
                int64_t start,
                int64_t stride,
                int64_t part_size,
                int64_t end,
                int64_t axis)
        : m_body_parameter(body_parameter),
          m_value(value),
          m_start(start),
          m_stride(stride),
          m_part_size(part_size),
          m_end(end),
          m_axis(axis) {}
    void add(std::shared_ptr<ov::op::util::SubGraphOp> subgraph) const override {
        subgraph->set_sliced_input(m_body_parameter, m_value, m_start, m_stride, m_part_size, m_end, m_axis);
    }

private:
    const std::shared_ptr<ov::op::v0::Parameter> m_body_parameter;
    const ov::Output<ov::Node> m_value;
    const int64_t m_start;
    const int64_t m_stride;
    const int64_t m_part_size;
    const int64_t m_end;
    const int64_t m_axis;
};

class SubgraphOutput {
public:
    SubgraphOutput() = default;
    virtual ~SubgraphOutput() = default;
    virtual void replace(ov::Output<ov::Node>& output) const = 0;
};

class SimpleSubgraphOutput : public SubgraphOutput {
public:
    explicit SimpleSubgraphOutput(const ov::Output<ov::Node>& output) : m_output(output) {}
    void replace(ov::Output<ov::Node>& output) const override {
        output.replace(m_output);
    }

private:
    const ov::Output<ov::Node> m_output;
};

class IterValueSubgraphOutput : public SubgraphOutput {
public:
    IterValueSubgraphOutput(const ov::Output<ov::Node>& value,
                            const std::shared_ptr<ov::op::util::SubGraphOp>& subgraph,
                            int64_t iteration)
        : m_value(value),
          m_subgraph(subgraph),
          m_iteration(iteration) {}
    void replace(ov::Output<ov::Node>& output) const override {
        auto new_output = m_subgraph->get_iter_value(m_value, m_iteration);
        output.replace(new_output);
    }

private:
    const ov::Output<ov::Node> m_value;
    const std::shared_ptr<ov::op::util::SubGraphOp> m_subgraph;
    const int64_t m_iteration;
};

// find Result nodes that are connected directly to Parameter node (except body condition output)
std::unordered_set<uint64_t> find_body_value_indexes_to_remove(const ov::ResultVector& body_results,
                                                               int64_t body_condition_output_idx) {
    std::unordered_set<uint64_t> body_value_indexes_to_remove;
    for (uint64_t body_value_index = 0; body_value_index < body_results.size(); ++body_value_index) {
        if (body_condition_output_idx == static_cast<int64_t>(body_value_index))
            continue;
        const auto parent_node = body_results[body_value_index]->input_values()[0].get_node();
        if (!dynamic_cast<ov::op::v0::Parameter*>(parent_node))
            continue;
        body_value_indexes_to_remove.emplace(body_value_index);
    }
    return body_value_indexes_to_remove;
}

// find Parameter nodes that have only Result nodes as consumers (except body condition output)
std::unordered_set<uint64_t> find_body_param_indexes_to_remove(
    const ov::ParameterVector& body_params,
    const std::shared_ptr<ov::op::v0::Result>& body_condition_output) {
    std::unordered_set<uint64_t> body_param_indexes_to_remove;
    for (uint64_t body_param_index = 0; body_param_index < body_params.size(); ++body_param_index) {
        const auto consumers = body_params[body_param_index]->get_output_target_inputs(0);
        if (std::all_of(consumers.begin(),
                        consumers.end(),
                        [&body_condition_output](const ov::Input<ov::Node>& consumer) {
                            const auto node = dynamic_cast<const ov::op::v0::Result*>(consumer.get_node());
                            if (!node)
                                return false;
                            // if it is not loop but TensorIterator there is no body condition output
                            if (!body_condition_output)
                                return true;
                            return body_condition_output->get_instance_id() != node->get_instance_id();
                        })) {
            body_param_indexes_to_remove.emplace(body_param_index);
        }
    }
    return body_param_indexes_to_remove;
}
}  // namespace

ov::pass::EliminateLoopInputsOutputs::EliminateLoopInputsOutputs() {
    MATCHER_SCOPE(EliminateLoopInputsOutputs);

    auto subgraph_label = wrap_type<ov::op::v5::Loop, ov::op::v0::TensorIterator>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto subgraph = as_type_ptr<op::util::SubGraphOp>(pattern_to_output.at(subgraph_label).get_node_shared_ptr());
        auto loop = as_type_ptr<ov::op::v5::Loop>(subgraph);

        // clone model since we will update it for the new loop
        auto body_model = subgraph->get_function()->clone();
        const auto body_params = body_model->get_parameters();
        const auto body_results = body_model->get_results();

        ov::op::v5::Loop::SpecialBodyPorts loop_special_ports;
        // if it is TensorIterator there is not body_condition_output_idx for it
        loop_special_ports.body_condition_output_idx = -1;
        if (loop) {
            loop_special_ports = loop->get_special_body_ports();
        }

        /* find Params and Results, that is connected together (except special ports)
         * If body param will be removed, it's all body values will be removed also
         */
        const auto body_value_indexes_to_remove =
            find_body_value_indexes_to_remove(body_results, loop_special_ports.body_condition_output_idx);

        // if it is TensorIterator there is not body_condition_output_idx for it
        std::shared_ptr<ov::op::v0::Result> body_condition_output;
        if (loop)
            body_condition_output = body_results[loop_special_ports.body_condition_output_idx];
        const auto body_param_indexes_to_remove = find_body_param_indexes_to_remove(body_params, body_condition_output);

        const auto subgraph_input_values = subgraph->input_values();
        const auto& trip_count = subgraph_input_values[0];
        const auto& exec_cond = subgraph_input_values[1];

        std::shared_ptr<op::util::SubGraphOp> new_subgraph;
        std::shared_ptr<ov::op::v5::Loop> new_loop;
        if (loop) {
            new_subgraph = make_shared<ov::op::v5::Loop>(trip_count, exec_cond);
            new_loop = as_type_ptr<ov::op::v5::Loop>(new_subgraph);
        } else {
            new_subgraph = make_shared<ov::op::v0::TensorIterator>();
        }
        new_subgraph->set_function(body_model);

        // save it to set up new loop special ports after removing body model results
        const auto& new_loop_body_condition_output =
            body_results[loop_special_ports.body_condition_output_idx];  // no for ov::op::v0::TensorIterator

        // collect data about loop inputs
        std::vector<std::shared_ptr<SubgraphInput>> new_subgraph_inputs;
        for (const auto& input_description : subgraph->get_input_descriptions()) {
            // if parameter removed, do not add input_description into new loop
            if (body_param_indexes_to_remove.find(input_description->m_body_parameter_index) !=
                body_param_indexes_to_remove.end()) {
                continue;
            }

            if (const auto merged_input_desc =
                    as_type_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>(input_description)) {
                /* if we have Parameter -> Result, input data is not changed in loop, and we can set up this
                 * input as "invariant" instead of making it "merged"
                 */
                if (body_value_indexes_to_remove.find(merged_input_desc->m_body_value_index) !=
                    body_value_indexes_to_remove.end()) {
                    new_subgraph_inputs.emplace_back(
                        std::make_shared<InvariantInput>(body_params[merged_input_desc->m_body_parameter_index],
                                                         subgraph_input_values[merged_input_desc->m_input_index]));
                } else {
                    new_subgraph_inputs.emplace_back(
                        std::make_shared<MergedInput>(body_params[merged_input_desc->m_body_parameter_index],
                                                      subgraph_input_values[merged_input_desc->m_input_index],
                                                      body_results[merged_input_desc->m_body_value_index]));
                }
            } else if (const auto invariant_input_desc =
                           as_type_ptr<ov::op::util::MultiSubGraphOp::InvariantInputDescription>(input_description)) {
                new_subgraph_inputs.emplace_back(
                    std::make_shared<InvariantInput>(body_params[invariant_input_desc->m_body_parameter_index],
                                                     subgraph_input_values[invariant_input_desc->m_input_index]));
            } else if (const auto sliced_input_desc =
                           as_type_ptr<ov::op::util::MultiSubGraphOp::SliceInputDescription>(input_description)) {
                new_subgraph_inputs.emplace_back(
                    std::make_shared<SlicedInput>(body_params[sliced_input_desc->m_body_parameter_index],
                                                  subgraph_input_values[sliced_input_desc->m_input_index],
                                                  sliced_input_desc->m_start,
                                                  sliced_input_desc->m_stride,
                                                  sliced_input_desc->m_part_size,
                                                  sliced_input_desc->m_end,
                                                  sliced_input_desc->m_axis));
            } else {
                /* unknown input description type
                 * this could only happen if new input description type was added after this transformation
                 * written
                 */
                return false;
            }
        }

        /*
         * This map {body_value index -> loop_input} is needed to simplify searching new loop input node in
         * output descriptors management step.
         */
        std::unordered_map<uint64_t, Output<Node>> body_subgraph_inputs;
        for (const auto& input_description : subgraph->get_input_descriptions()) {
            const auto merged_input_desc =
                as_type_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>(input_description);
            if (!merged_input_desc)
                continue;
            body_subgraph_inputs.emplace(merged_input_desc->m_body_value_index,
                                         subgraph_input_values[merged_input_desc->m_input_index]);
        }

        // collect data about loop outputs
        std::vector<std::shared_ptr<SubgraphOutput>> new_subgraph_outputs;
        int64_t iteration = -1;
        for (const auto& output_description : subgraph->get_output_descriptions()) {
            iteration = -1;
            if (const auto body_output_desc =
                    as_type_ptr<ov::op::util::MultiSubGraphOp::BodyOutputDescription>(output_description)) {
                iteration = body_output_desc->m_iteration;
            }
            if (body_value_indexes_to_remove.find(output_description->m_body_value_index) !=
                body_value_indexes_to_remove.end()) {
                new_subgraph_outputs.emplace_back(std::make_shared<SimpleSubgraphOutput>(
                    body_subgraph_inputs[output_description->m_body_value_index]));
            } else {
                new_subgraph_outputs.emplace_back(
                    std::make_shared<IterValueSubgraphOutput>(body_results[output_description->m_body_value_index],
                                                              new_subgraph,
                                                              iteration));
            }
        }

        // remove Params, Results from new loop body
        for (const auto result_idx : body_value_indexes_to_remove) {
            body_model->remove_result(body_results[result_idx]);
        }
        for (const auto param_idx : body_param_indexes_to_remove) {
            body_model->remove_parameter(body_params[param_idx]);
        }

        // set up new loop special body ports (if it is Loop)
        if (new_loop) {
            const int64_t in_body_condition_output_idx = body_model->get_result_index(new_loop_body_condition_output);
            const ov::op::v5::Loop::SpecialBodyPorts ports(loop_special_ports.current_iteration_input_idx,
                                                           in_body_condition_output_idx);
            new_loop->set_special_body_ports(ports);
        }

        // connect new loop inputs with body inputs
        for (const auto& input : new_subgraph_inputs) {
            input->add(new_subgraph);
        }

        // connect new loop outputs with body outputs
        auto loop_outputs = subgraph->outputs();
        for (size_t i = 0; i < loop_outputs.size(); ++i) {
            new_subgraph_outputs[i]->replace(loop_outputs[i]);
        }

        ov::copy_runtime_info(subgraph, new_subgraph);
        new_subgraph->set_friendly_name(subgraph->get_friendly_name());

        return true;
    };

    auto m = make_shared<Matcher>(subgraph_label, matcher_name);
    this->register_matcher(m, callback);
}
