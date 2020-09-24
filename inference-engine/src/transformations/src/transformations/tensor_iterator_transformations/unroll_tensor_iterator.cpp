// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/tensor_iterator_transformations/unroll_tensor_iterator.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/graph_util.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::UnrollTensorIterator, "UnrollTensorIterator", 0);

ngraph::pass::UnrollTensorIterator::UnrollTensorIterator() : MatcherPass() {
    auto tensor_iterator = ngraph::pattern::wrap_type<ngraph::opset4::TensorIterator>();
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator>(m.get_match_root());
        if (!ti) {
            return false;
        }

        const auto function = ti->get_body();
        auto num_iter = ti->get_num_iterations();

        // negative value means inconsistent TI
        if (num_iter <= -1) {
            return false;
        }

        // Create copies of the TensorIterator body, the number of copies is equal to the number of iterations.
        // Assign names to the created layers.
        std::vector<std::shared_ptr<ngraph::Function>> body_functions(num_iter);
        for (uint64_t idx = 0; idx < num_iter; ++idx) {
            body_functions[idx] = clone_function(*function);
            for (auto &node : body_functions[idx]->get_ops()) {
                node->set_friendly_name(ti->get_friendly_name() + "/" + std::to_string(idx + 1) + "/" + node->get_friendly_name());
                copy_runtime_info(ti, node);
            }
        }

        // Port map : inputs and back edges
        for (const auto& desc : ti->get_input_descriptions()) {
            const std::string& type_name = desc->get_type_info().name;

            if (type_name == "SliceInputDescription") {
                auto input_desc = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator::SliceInputDescription>(desc);
                if (!input_desc) {
                    return false;
                }

                // Connect the sliced input (layer before the input) to the Split layer and connect
                // the corresponding Split output to the corresponding copy of the body.
                // If the number of iterations is 1, then the Split is not needed.

                auto in_data = ti->input_values()[input_desc->m_input_index];
                const auto const_axis = opset4::Constant::create(element::i64, Shape{}, {input_desc->m_axis});

                if (num_iter > 1) {
                    auto split = std::make_shared<ngraph::opset4::Split>(in_data, const_axis, num_iter);
                    copy_runtime_info(ti, split);
                    auto stride = input_desc->m_stride;
                    // connect to the body
                    for (uint64_t j = 0; j < num_iter; j++) {
                        auto idx = stride > 0 ? j : num_iter - j - 1;
                        auto param = body_functions[j]->get_parameters()[input_desc->m_body_parameter_index];
                        for (auto &output : param->outputs()) {
                            output.replace(split->output(idx));
                        }
                    }
                } else {
                    // connect to the body
                    auto param = body_functions[0]->get_parameters()[input_desc->m_body_parameter_index];
                    for (auto &output : param->outputs()) {
                        output.replace(in_data);
                    }
                }
            } else if (type_name == "MergedInputDescription") {
                auto input_desc = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator::MergedInputDescription>(desc);
                if (!input_desc) {
                    return false;
                }

                // Connect the input to the corresponding copy of the body.
                auto in_data = ti->input_values()[input_desc->m_input_index].get_node_shared_ptr();
                auto param = body_functions[0]->get_parameters()[input_desc->m_body_parameter_index];
                for (auto &output : param->outputs()) {
                    output.replace(in_data);
                }

                // Back-edge processing. Connect the copies of the body to each other.
                for (uint64_t j = 1; j < num_iter; j++) {
                    auto cur_param = body_functions[j]->get_parameters()[input_desc->m_body_parameter_index];
                    auto prev_val = body_functions[j - 1]->get_results()[input_desc->m_body_value_index];
                    for (auto &output : cur_param->outputs()) {
                        output.replace(prev_val->get_input_source_output(0));
                    }
                }
            } else if (type_name == "InvariantInputDescription") {
                auto input_desc = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator::InvariantInputDescription>(
                        desc);
                if (!input_desc) {
                    return false;
                }

                // Connect the input to the corresponding copy of the body.
                auto in_data = ti->input_values()[input_desc->m_input_index].get_node_shared_ptr();
                for (uint64_t j = 0; j < num_iter; j++) {
                    auto param = body_functions[j]->get_parameters()[input_desc->m_body_parameter_index];
                    for (auto &output : param->outputs()) {
                        output.replace(in_data);
                    }
                }
            } else {
                // "Incorrect type of the input description.";
                return false;
            }
        }

        // Port map: outputs
        for (const auto& desc : ti->get_output_descriptions()) {
            std::string type_name = desc->get_type_info().name;
            if (type_name == "ConcatOutputDescription") {
                auto output_desc = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator::ConcatOutputDescription>(desc);
                if (!output_desc) {
                    return false;
                }

                // Connect corresponding outputs (layers before Result op) of each copy of the body to Concat layer.
                // Connect the Concat to corresponding output of TensorIterator.
                // If the number of iterations is 1, then the Concat is not needed.

                if (num_iter > 1) {
                    ngraph::OutputVector to_concat(num_iter);
                    auto stride = output_desc->m_stride;

                    // Connect outputs of the bodies to the Concat layer
                    for (uint64_t j = 0; j < num_iter; j++) {
                        auto idx = stride > 0 ? j : num_iter - j - 1;
                        std::shared_ptr<opset4::Result> result = body_functions[idx]->get_results()[output_desc->m_body_value_index];
                        auto input_to_res = result->get_input_source_output(0);
                        to_concat[j] = input_to_res;
                    }
                    auto concat = std::make_shared<ngraph::opset4::Concat>(to_concat, output_desc->m_axis);
                    copy_runtime_info(ti, concat);

                    // connect the Concat layer to the corresponding TI outputs
                    concat->output(0).get_tensor().set_name(
                            op::util::create_ie_output_name(ti->output(output_desc->m_output_index)));
                    for (auto &input : ti->output(output_desc->m_output_index).get_target_inputs()) {
                        input.replace_source_output(concat);
                    }
                } else {
                    // Connect outputs of the bodies to the corresponding TI outputs
                    std::shared_ptr<opset4::Result> result = body_functions[0]->get_results()[output_desc->m_body_value_index];
                    auto input_to_res = result->get_input_source_output(0);
                    for (auto &input : ti->output(output_desc->m_output_index).get_target_inputs()) {
                        input.replace_source_output(input_to_res);
                    }
                }
            } else if (type_name == "BodyOutputDescription") {
                auto output_desc = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator::BodyOutputDescription>(desc);
                if (!output_desc) {
                    return false;
                }

                // Connect outputs of the bodies to the corresponding TI outputs
                auto iter = output_desc->m_iteration;
                iter = iter >= 0? iter: num_iter - 1;
                std::shared_ptr<opset4::Result> result = body_functions[iter]->get_results()[output_desc->m_body_value_index];
                const auto& in_value = result->input_value(0);

                in_value.get_tensor().set_name(op::util::create_ie_output_name(ti->output(output_desc->m_output_index)));
                for (const auto &input : ti->output(output_desc->m_output_index).get_target_inputs()) {
                    input.replace_source_output(result->get_input_source_output(0));
                }
            } else {
                // "Incorrect type of the output description."
                return false;
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "UnrollTensorIterator");
    register_matcher(m, callback);
}