// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/propagate_precision.hpp"

#include <assert.h>
#include <memory>
#include "openvino/core/except.hpp"
#include "ie_ngraph_utils.hpp"
#include "snippets/itt.hpp"
#include "ngraph/rt_info.hpp"

ngraph::snippets::pass::PropagatePrecision::PropagatePrecision(
    const ov::element::Type supported_precision,
    const std::shared_ptr<const TargetMachine>& target_machine) : supported_precision(supported_precision), target_machine(target_machine) {
}

bool ngraph::snippets::pass::PropagatePrecision::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(PropagatePrecision);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::PropagatePrecision")

    std::unordered_map<std::shared_ptr<ngraph::opset1::Result>, element::Type> result_types;
    auto results = f->get_results();
    for (auto& result : results) {
        result_types.emplace(result, result->get_input_source_output(0).get_element_type());
    }

    for (const auto& op : f->get_ordered_ops()) {
        if (ngraph::is_type<opset1::Constant>(op)) {
            continue;
        }

        auto type_info = op->get_type_info();
        if (!target_machine->has(type_info)) {
            throw ov::Exception(
                "operation '" + std::string(type_info.version_id) + "::" + std::string(type_info.name) +
                "' was not found in target machine");
        }

        const auto supported_precisions = target_machine->get_supported_precisions(type_info);
        if (supported_precisions.empty()) {
            continue;
        }

        bool alligned_inputs = true;
        for (const auto& input : op->inputs()) {
            if (!ov::is_type<ngraph::snippets::op::ConvertSaturation>(input.get_source_output().get_node())) {
                alligned_inputs = false;
                break;
            }
        }

        std::vector<InferenceEngine::Precision> input_precisions;
        for (const auto& input : op->inputs()) {
            const auto parent_input = alligned_inputs ? input.get_source_output().get_node()->input(0) : input;

            const auto input_precision = parent_input.get_source_output().get_element_type();
            const auto input_precision_ie = InferenceEngine::details::convertPrecision(input_precision);
            input_precisions.push_back(input_precision_ie);
        }

        assert(std::all_of(
            supported_precisions.begin(),
            supported_precisions.end(),
            [&input_precisions](const std::vector<InferenceEngine::Precision>& precisions) {
                return precisions.size() == input_precisions.size();
            }) && "input precisions count is not equal for supported precisions");

        // if possible remove alligned input convertions
        if (alligned_inputs &&
            std::any_of(
                supported_precisions.begin(),
                supported_precisions.end(),
                [&input_precisions](const std::vector<InferenceEngine::Precision>& precisions) {
                    return precisions == input_precisions;
                })) {
            std::vector<element::Type> original_types;
            for (const auto& output : op->outputs()) {
                original_types.push_back(output.get_element_type());
            }

            for (auto i = 0; i < op->get_input_size(); ++i) {
                const auto convert = op->get_input_node_shared_ptr(i);
                assert(ov::is_type<ngraph::snippets::op::ConvertSaturation>(convert));
                op->set_argument(i, convert->input(0).get_source_output());
                //convert->input(0).get_source_output().replace(op);
            }

            op->validate_and_infer_types();

            auto insert_final_convert = false;
            for (auto i = 0ull; i < op->get_output_size(); ++i) {
                if (original_types[i] != op->output(i).get_element_type()) {
                    insert_final_convert = true;
                    break;
                }
            }

            if (insert_final_convert) {
                for (auto i = 0ull; i < op->get_output_size(); ++i) {
                    const auto& op_output = op->output(i);
                    for (const auto& input : op_output.get_target_inputs()) {
                        const auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                            op_output,
                            original_types[i]);
                        input.replace_source_output(convert->output(0));
                    }
                }
            }
            continue;
        }

        // if possible then convert precisions to supported
        if (!supported_precisions.empty() &&
            !std::any_of(
                supported_precisions.begin(),
                supported_precisions.end(),
                [&input_precisions](const std::vector<InferenceEngine::Precision>& precisions) {
                    return precisions == input_precisions;
                })) {
            auto precisions = get_precisions(input_precisions,
                                             supported_precisions,
                                             InferenceEngine::details::convertPrecision(supported_precision));
            if (precisions.empty()) {
                throw ov::Exception(
                    "there are no supported precisions for operation '" +
                    std::string(type_info.version_id) + "::" +
                    std::string(type_info.name) + "'");
            }

            for (auto i = 0; i < op->get_input_size(); ++i) {
                const auto& input = op->input(i);
                const auto& precision = precisions[i];
                auto parent_output = input.get_source_output();
                auto const element_type = InferenceEngine::details::convertPrecision(precision);
                if (parent_output.get_element_type() != element_type) {
                    auto existing_convert = ngraph::as_type<ngraph::snippets::op::ConvertSaturation>(
                        parent_output.get_node());
                    if (existing_convert == nullptr) {
                        auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                            parent_output,
                            element_type);
                        parent_output.remove_target_input(input);
                        input.replace_source_output(convert->output(0));
                    } else {
                        auto parent_output = existing_convert->get_input_source_output(0);
                        if (element_type == parent_output.get_element_type()) {
                            existing_convert->output(0).replace(parent_output);
                        } else {
                            auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                                existing_convert->get_input_node_shared_ptr(0),
                                element_type);
                            replace_node(existing_convert->shared_from_this(), convert);
                            copy_runtime_info(existing_convert->shared_from_this(), convert);
                        }
                    }
                }
            }
            op->validate_and_infer_types();
        }
    }

    for (auto it = result_types.begin(); it != result_types.end(); ++it) {
        const auto result = it->first;
        const auto actual_type = result->get_input_source_output(0).get_element_type();
        const auto expected_type = it->second;
        if (actual_type != it->second) {
            auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                result->get_input_node_shared_ptr(0),
                expected_type);
            result->get_input_source_output(0).remove_target_input(result->input(0));
            result->input(0).replace_source_output(convert->output(0));
        }
    }

    return false;
}

std::vector<InferenceEngine::Precision> ngraph::snippets::pass::PropagatePrecision::get_precisions(
    const std::vector<InferenceEngine::Precision>& input_precisions,
    const std::set<std::vector<InferenceEngine::Precision>>& supported_precisions_pack,
    const InferenceEngine::Precision& base_precision) noexcept {
    bool was_found = false;
    for (const auto& supported_precisions : supported_precisions_pack) {
        for (auto i = 0ull; i < supported_precisions.size(); ++i) {
            const auto& supported_precision = supported_precisions[i];
            const auto& input_precision = input_precisions[i];
            if ((supported_precision.is_float() != input_precision.is_float()) ||
                (input_precision.bitsSize() > supported_precision.bitsSize())) {
                was_found = false;
                break;
            }

            was_found = true;
        }
        if (was_found) {
            return supported_precisions;
        }
    }
    for (const auto& supported_precisions : supported_precisions_pack) {
        if (supported_precisions[0] == base_precision) {
            return supported_precisions;
        }
    }
    return {};
}
