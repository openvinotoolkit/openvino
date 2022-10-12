// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/low_latency.hpp"

#include <memory>
#include <ngraph/log.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <openvino/cc/pass/itt.hpp>

NGRAPH_SUPPRESS_DEPRECATED_START
NGRAPH_RTTI_DEFINITION(ngraph::pass::LowLatency, "LowLatency", 0);

using namespace std;

namespace {
string generate_variable_name(const string& op_name, const string& param_name, int64_t variable_idx) {
    return op_name + "/" + param_name + "/" + "variable_" + to_string(variable_idx);
}

}  // namespace
ngraph::pass::LowLatency::LowLatency() {
    MATCHER_SCOPE(LowLatency);
    auto tensor_iterator = ov::pass::pattern::wrap_type<opset6::TensorIterator, opset6::Loop>();
    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        const auto& sub_graph_op = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(m.get_match_root());
        if (!sub_graph_op) {
            return false;
        }

        if (const auto& loop = std::dynamic_pointer_cast<opset6::Loop>(sub_graph_op)) {
            const auto& trip_count = std::dynamic_pointer_cast<opset6::Constant>(loop->get_input_node_shared_ptr(0));
            const auto& num_iter = loop->get_num_iterations();
            if (trip_count && num_iter > 0 && trip_count->get_output_target_inputs(0).size() == 1) {
                auto single_iter = std::make_shared<opset6::Constant>(ov::element::i64, Shape{}, 1);
                replace_node(trip_count, single_iter);
            } else {
                // count of iterations is dynamic;
                return false;
            }
        }
        // Mark the TI layer to be unrolled. Enable unconditional ti unrolling for all plugins.
        auto& rt_info = sub_graph_op->get_rt_info();
        rt_info["UNROLL_TI"] = int64_t(1);

        int64_t variable_id = 0;
        std::vector<std::shared_ptr<ngraph::op::Sink>> assigns;
        const auto& func = sub_graph_op->get_function();
        for (const auto& in : sub_graph_op->get_input_descriptions()) {
            // Process all back edges
            if (const auto& merged_in =
                    std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::MergedInputDescription>(in)) {
                // Insert ReadValue nodes: Parameter -> (new ReadValue) -> consumers
                const auto& inputs_to =
                    func->get_parameters().at(merged_in->m_body_parameter_index)->get_output_target_inputs(0);
                const std::string variable_name(generate_variable_name(
                    sub_graph_op->get_friendly_name(),
                    func->get_parameters().at(merged_in->m_body_parameter_index)->get_friendly_name(),
                    variable_id));
                auto variable = std::make_shared<Variable>(
                    VariableInfo{ov::PartialShape::dynamic(), element::dynamic, variable_name});
                auto read_value =
                    std::make_shared<opset6::ReadValue>(func->get_parameters().at(merged_in->m_body_parameter_index),
                                                        variable);
                read_value->set_friendly_name(variable_name);
                for (const auto& input_to : inputs_to) {
                    input_to.replace_source_output(read_value->output(0));
                }

                // insert Assign nodes: provider -> (new Assign) -> Result
                const auto res = func->get_results().at(merged_in->m_body_value_index);
                auto assign = std::make_shared<opset6::Assign>(res->input_value(0), variable);
                // control dependency so that ReadValue is processed before Assign
                assign->add_control_dependency(read_value);
                assigns.emplace_back(assign);
            }
            variable_id++;
        }
        // save Assign in the func so that it gets into graph traversals and isn't deleted.
        func->add_sinks(assigns);
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(tensor_iterator, matcher_name);
    register_matcher(m, callback);
}
NGRAPH_SUPPRESS_DEPRECATED_END

namespace {

void UnrollSingleIteration(const shared_ptr<ngraph::op::util::SubGraphOp>& sub_graph_op,
                           const shared_ptr<ov::Model>& outer_f) {
    using namespace ngraph::opset7;

    const auto& params = sub_graph_op->get_function()->get_parameters();
    const auto& results = sub_graph_op->get_function()->get_results();

    // before: Layer1 -> TI [input -> bodyParameter -> Layer2 -> ...]
    // after:  Layer1 -> Layer2 ->...
    for (const auto& in : sub_graph_op->get_input_descriptions()) {
        const auto& connect_to = sub_graph_op->get_input_source_output(in->m_input_index);
        for (auto& output : params.at(in->m_body_parameter_index)->outputs()) {
            output.replace(connect_to);
        }
    }

    // before: TI [...-> Layer1 -> Result -> output] -> Layer2 -> ...
    // after:  ...-> Layer1 -> Layer2 -> ...
    ov::NodeVector new_ops;
    for (const auto& out : sub_graph_op->get_output_descriptions()) {
        const auto& connect_to = results.at(out->m_body_value_index)->get_input_source_output(0);
        for (auto& input_to : sub_graph_op->output(out->m_output_index).get_target_inputs()) {
            // create IE output name
            std::string out_name = sub_graph_op->get_friendly_name();
            if (sub_graph_op->get_output_size() != 1)
                out_name += "." + std::to_string(out->m_output_index);

            // IECompatibility: insert identity (Unsqueeze + Squeeze) to store the TensorIterator
            // output names
            auto axis_1 = Constant::create(ov::element::i64, ov::Shape{1}, {1});
            auto identity_1 = std::make_shared<Unsqueeze>(connect_to, axis_1);
            auto identity_2 = std::make_shared<Squeeze>(identity_1, axis_1);
            identity_2->set_friendly_name(out_name);
            new_ops.push_back(identity_1);
            new_ops.push_back(identity_2);

            input_to.replace_source_output(identity_2);
        }
    }
    outer_f->add_sinks(sub_graph_op->get_function()->get_sinks());
    ngraph::copy_runtime_info(sub_graph_op, sub_graph_op->get_function()->get_ops());
    ngraph::copy_runtime_info(sub_graph_op, new_ops);
}

ngraph::Output<ngraph::Node> create_init_subgraph(const shared_ptr<ngraph::op::util::SubGraphOp>& sub_graph_op,
                                                  const ngraph::Output<ngraph::Node>& in_node) {
    using namespace ngraph::opset7;

    auto const_zero = make_shared<Constant>(in_node.get_element_type(), ov::Shape{1}, 0);
    auto shape_of = make_shared<ShapeOf>(in_node);
    auto broadcast = make_shared<Broadcast>(const_zero, shape_of);
    copy_runtime_info(sub_graph_op, {const_zero, shape_of, broadcast});
    return broadcast->output(0);
}

}  // namespace

bool ov::pass::LowLatency2::run_on_model(const shared_ptr<Model>& f) {
    using namespace ngraph::opset7;

    RUN_ON_MODEL_SCOPE(LowLatency2);
    ngraph::SinkVector assigns;
    for (const auto& op : f->get_ordered_ops()) {
        if (const auto& sub_graph_op = dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(op)) {
            int64_t variable_id = 0;
            const auto& func = sub_graph_op->get_function();
            const auto& params = func->get_parameters();
            for (const auto& in : sub_graph_op->get_input_descriptions()) {
                // Process all back edges
                if (const auto& merged_in =
                        dynamic_pointer_cast<ngraph::op::util::SubGraphOp::MergedInputDescription>(in)) {
                    // create new Variable
                    const string& param_name = params.at(merged_in->m_body_parameter_index)->get_friendly_name();
                    const string& var_name =
                        generate_variable_name(sub_graph_op->get_friendly_name(), param_name, variable_id);

                    const auto& input = sub_graph_op->input(merged_in->m_input_index);
                    if (std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(
                            input.get_source_output().get_node_shared_ptr()) != nullptr) {
                        NGRAPH_DEBUG << "LowLatency2 transformation cannot be applied because the "
                                     << "ReadValue node is already an input to the TensorIterator."
                                     << "LowLatency2 transformation may have already been applied, please "
                                     << "do not call it more then once.";
                        return false;
                    }

                    const auto& param =
                        sub_graph_op->get_function()->get_parameters().at(merged_in->m_body_parameter_index);
                    for (const auto& in_to : param->output(0).get_target_inputs()) {
                        if (dynamic_cast<ngraph::op::ReadValueBase*>(in_to.get_node()) != nullptr) {
                            NGRAPH_DEBUG << "LowLatency2 transformation cannot be applied because the "
                                         << "ReadValue node is already inside the TensorIterator. "
                                         << "LowLatency transformation may have been applied, please do "
                                         << "not call LowLatency2 after LowLatency.";
                            return false;
                        }
                    }

                    ngraph::VariableInfo var_info{PartialShape::dynamic(), element::dynamic, var_name};
                    auto variable = make_shared<ngraph::Variable>(var_info);

                    // insert ReadValue
                    // Layers -> [new op: ReadValue] -> Subgraph operation
                    Output<Node> read_value_in = input.get_source_output();
                    if (m_use_const_initializer) {
                        read_value_in = create_init_subgraph(sub_graph_op, read_value_in);
                    }
                    auto read_value = make_shared<ReadValue>(read_value_in, variable);
                    input.replace_source_output(read_value->output(0));
                    read_value->set_friendly_name(var_name);
                    ngraph::copy_runtime_info(sub_graph_op, read_value);

                    /* insert Assign
                    // Subgraph operation -> [new op: Assign]
                    //                    \
                    //                      ---> Layers -> ...
                    */
                    const auto& out_desc = sub_graph_op->get_output_descriptions();
                    bool is_output_exist = std::any_of(
                        out_desc.begin(),
                        out_desc.end(),
                        [&merged_in](const std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>& out) {
                            return out->m_body_value_index == merged_in->m_body_value_index;
                        });
                    // Create new output if it doesn't exist.
                    if (!is_output_exist) {
                        sub_graph_op->get_iter_value(func->get_results().at(merged_in->m_body_value_index));
                    }
                    for (const auto& out : sub_graph_op->get_output_descriptions()) {
                        if (out->m_body_value_index == merged_in->m_body_value_index) {
                            auto assign = make_shared<Assign>(sub_graph_op->output(out->m_output_index), variable);
                            copy_runtime_info(sub_graph_op, assign);
                            // control dependency so that ReadValue is processed before Assign
                            assign->add_control_dependency(read_value);
                            assigns.emplace_back(assign);
                            break;
                        }
                    }
                }

                variable_id++;
            }

            if (sub_graph_op->get_num_iterations() == 1) {
                UnrollSingleIteration(sub_graph_op, f);
            }
        }
    }
    f->add_sinks(assigns);
    return true;
}
