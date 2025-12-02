// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tuple_unpack_replacer.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

PrimTupleUnpackReplacer::PrimTupleUnpackReplacer() {
    auto tuple_unpack =
        ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>(fw_node_predicate({"prim::TupleUnpack"}));

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto tuple_unpack = m.get_match_root();
        OutputVector outputs;
        auto input_node = tuple_unpack->get_input_node_shared_ptr(0);
        if (cast_fw_node(input_node, "prim::TupleConstruct")) {
            for (const auto& input : input_node->inputs()) {
                const auto& out = input.get_source_output();
                outputs.push_back(out);
            }
            replace_node(tuple_unpack, outputs);

            return true;
        } else if (ov::as_type_ptr<v0::Constant>(input_node)) {
            // tuple might have been merged as a single constant
            auto axis_zero = v0::Constant::create(element::i32, Shape{}, {0});
            auto split = std::make_shared<v1::Split>(input_node, axis_zero, tuple_unpack->outputs().size());
            for (size_t i = 0; i < split->get_output_size(); ++i) {
                auto squeeze = std::make_shared<v15::Squeeze>(split->output(i), axis_zero);
                replace_output_update_name(tuple_unpack->output(i), squeeze);
            }
            return true;
        } else if (input_node->get_rt_info().count("__torch_tuple_unpackable__")) {
            // This case is produced by inlined_extension
            input_node->get_rt_info().erase("__torch_tuple_unpackable__");
            // remove TupleUnpack just bypassing it with all outputs from a custom operation which returns tuple
            replace_node(tuple_unpack, input_node->outputs());
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(tuple_unpack,
                                                          "ov::frontend::pytorch::pass::PrimTupleUnpackReplacer");
    this->register_matcher(m, callback);
};

bool TupleUnpackInBodyReplacer::run_on_model(const std::shared_ptr<Model>& model) {
    bool result = false;
    for (auto& op : model->get_ordered_ops()) {
        const auto if_op = as_type_ptr<v8::If>(op);
        if (if_op) {
            for (size_t i = 1; i < if_op->get_input_size(); i++) {
                auto input = if_op->input_value(i);
                auto tuple_construct = ov::as_type_ptr<ov::frontend::pytorch::PtFrameworkNode>(
                    cast_fw_node(input.get_node_shared_ptr(), "prim::TupleConstruct"));
                if (!tuple_construct) {
                    continue;
                }
                int then_body_idx = -1;
                int else_body_idx = -1;
                auto then_descs = if_op->get_input_descriptions(v8::If::THEN_BODY_INDEX);
                auto else_descs = if_op->get_input_descriptions(v8::If::ELSE_BODY_INDEX);
                for (auto& inp_desc : then_descs) {
                    if (inp_desc->m_input_index == i) {
                        if (then_body_idx != -1) {
                            add_exception_to_fw_node(
                                tuple_construct,
                                "Unexpected: TupleConstruct output is used in body more then once.");
                        } else {
                            then_body_idx = static_cast<int>(inp_desc->m_body_parameter_index);
                        }
                    }
                }
                for (auto& inp_desc : else_descs) {
                    if (inp_desc->m_input_index == i) {
                        if (else_body_idx != -1) {
                            add_exception_to_fw_node(
                                tuple_construct,
                                "Unexpected: TupleConstruct output is used in body more then once.");
                        } else {
                            else_body_idx = static_cast<int>(inp_desc->m_body_parameter_index);
                        }
                    }
                }
                auto new_if = std::make_shared<v8::If>(if_op->input_value(0));
                auto then_body = if_op->get_function(v8::If::THEN_BODY_INDEX);
                auto else_body = if_op->get_function(v8::If::ELSE_BODY_INDEX);
                ov::ParameterVector new_then_params;
                ov::ParameterVector new_else_params;
                if (then_body_idx != -1) {
                    auto then_param = then_body->get_parameters().at(then_body_idx);
                    ov::OutputVector new_tc_inputs;
                    for (size_t i = 0; i < tuple_construct->get_input_size(); i++) {
                        auto new_param = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic());
                        new_then_params.push_back(new_param);
                        new_tc_inputs.push_back(new_param);
                    }
                    auto new_tc =
                        std::make_shared<ov::frontend::pytorch::PtFrameworkNode>(tuple_construct->get_decoder(),
                                                                                 new_tc_inputs,
                                                                                 1);
                    then_body->add_parameters(new_then_params);
                    then_body->remove_parameter(then_param);
                    then_param->output(0).replace(new_tc->output(0));
                }
                if (else_body_idx != -1) {
                    auto else_param = else_body->get_parameters().at(else_body_idx);
                    ov::OutputVector new_tc_inputs;
                    for (size_t i = 0; i < tuple_construct->get_input_size(); i++) {
                        auto new_param = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic());
                        new_else_params.push_back(new_param);
                        new_tc_inputs.push_back(new_param);
                    }
                    auto new_tc =
                        std::make_shared<ov::frontend::pytorch::PtFrameworkNode>(tuple_construct->get_decoder(),
                                                                                 new_tc_inputs,
                                                                                 1);
                    else_body->add_parameters(new_else_params);
                    else_body->remove_parameter(else_param);
                    else_param->output(0).replace(new_tc->output(0));
                }
                new_if->set_function(v8::If::THEN_BODY_INDEX, then_body);
                new_if->set_function(v8::If::ELSE_BODY_INDEX, else_body);
                new_if->set_output_size(if_op->get_output_size());
                new_if->set_output_descriptions(v8::If::THEN_BODY_INDEX,
                                                if_op->get_output_descriptions(v8::If::THEN_BODY_INDEX));
                new_if->set_output_descriptions(v8::If::ELSE_BODY_INDEX,
                                                if_op->get_output_descriptions(v8::If::ELSE_BODY_INDEX));

                // create new If inputs
                std::vector<std::pair<int, int>> inputs_mapping(if_op->get_input_size(), {-1, -1});
                for (auto& inp_desc : then_descs) {
                    inputs_mapping[inp_desc->m_input_index].first = static_cast<int>(inp_desc->m_body_parameter_index);
                }
                for (auto& inp_desc : else_descs) {
                    inputs_mapping[inp_desc->m_input_index].second = static_cast<int>(inp_desc->m_body_parameter_index);
                }
                for (size_t j = 0; j < inputs_mapping.size(); j++) {
                    if (j == i)
                        continue;
                    int then_p_idx = inputs_mapping[j].first;
                    if (then_p_idx > then_body_idx && then_body_idx != -1)
                        then_p_idx--;
                    int else_p_idx = inputs_mapping[j].second;
                    if (else_p_idx > else_body_idx && else_body_idx != -1)
                        else_p_idx--;
                    const auto& then_p = then_p_idx == -1 ? nullptr : then_body->get_parameters()[then_p_idx];
                    const auto& else_p = else_p_idx == -1 ? nullptr : else_body->get_parameters()[else_p_idx];
                    if (then_p || else_p)
                        new_if->set_invariant_inputs(if_op->input_value(j), {then_p, else_p});
                }
                for (size_t j = 0; j < tuple_construct->get_input_size(); j++) {
                    ParameterVector body_inps;
                    if (then_body_idx != -1) {
                        FRONT_END_GENERAL_CHECK(j < new_then_params.size(), "Unexpected number of Parameters.");
                        body_inps.push_back(new_then_params[j]);
                    } else {
                        body_inps.push_back(nullptr);
                    }
                    if (else_body_idx != -1) {
                        FRONT_END_GENERAL_CHECK(j < new_else_params.size(), "Unexpected number of Parameters.");
                        body_inps.push_back(new_else_params[j]);
                    } else {
                        body_inps.push_back(nullptr);
                    }
                    new_if->set_invariant_inputs(tuple_construct->input_value(j), body_inps);
                }
                new_if->set_friendly_name(if_op->get_friendly_name());
                replace_node(if_op, new_if);
                new_if->validate_and_infer_types();
                op = std::dynamic_pointer_cast<Node>(new_if);
                result = true;
                break;
            }
        }
        if (const auto multiSubGraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op)) {
            for (size_t i = 0; i < multiSubGraph->get_internal_subgraphs_size(); i++)
                result = run_on_model(multiSubGraph->get_function(i)) || result;
        }
    }

    return result;
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
