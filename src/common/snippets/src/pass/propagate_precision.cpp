// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/propagate_precision.hpp"

#include "ov_ops/type_relaxed.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "transformations/utils/utils.hpp"

#include <assert.h>
#include <memory>


ov::snippets::pass::PropagatePrecision::PropagatePrecision(
    const std::shared_ptr<const TargetMachine>& target_machine) : target_machine(target_machine) {
}

bool ov::snippets::pass::PropagatePrecision::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(PropagatePrecision);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::PropagatePrecision")

    std::unordered_map<std::shared_ptr<ov::opset1::Result>, element::Type> result_types;
    auto results = f->get_results();
    for (auto& result : results) {
        result_types.emplace(result, result->get_input_element_type(0));
    }

    bool was_updated = false;
    for (const auto& op : f->get_ordered_ops()) {
        ov::op::util::process_subgraph(*this, op);

        auto type_info = op->get_type_info();
        auto exec = target_machine->get_supported_precisions(type_info);
        const auto& supported_precisions = exec(op);

        if (supported_precisions.empty()) {
            continue;
        }

        // There are two operation types which break precision propagation:
        //   1) Existing convertion operations. Solution: remove convertion
        //      operation before general algo
        //   2) Type relaxed based operations. Will be resolved by snippet opset.

        for (const auto& input : op->inputs()) {
            const auto convert = ov::as_type<snippets::op::ConvertSaturation>(input.get_source_output().get_node());
            if (convert == nullptr) {
                continue;
            }

            const auto precision_before = convert->get_input_element_type(0);
            const auto precision_after = convert->get_output_element_type(0);
            if (can_be_removed(precision_before, precision_after, precision_before)) {
                op->set_argument(input.get_index(), convert->input(0).get_source_output());
                was_updated = true;
            }
        }

        std::vector<element::Type> input_precisions;
        for (const auto& input : op->inputs()) {
            const auto input_precision = input.get_source_output().get_element_type();
            input_precisions.push_back(input_precision);
        }

        assert(std::all_of(
            supported_precisions.begin(),
            supported_precisions.end(),
            [&input_precisions](const std::vector<element::Type>& precisions) {
                return precisions.size() == input_precisions.size();
            }) && "input precisions count is not equal for supported precisions");

        // update input precisions
        // if possible then convert precisions to supported
        if (!supported_precisions.empty() &&
            std::all_of(
                supported_precisions.begin(),
                supported_precisions.end(),
                [&input_precisions](const std::vector<element::Type>& precisions) {
                    return precisions != input_precisions;
                })) {
            auto precisions = get_precisions(input_precisions,
                                             supported_precisions);
            OPENVINO_ASSERT(
                !precisions.empty(),
                "there are no supported precisions for operation '" + std::string(type_info.version_id) + "::" + std::string(type_info.name) + "'");

            auto find_convert = [](
                const ov::Output<ov::Node> parent_output,
                const ov::element::Type convert_type) -> snippets::op::ConvertSaturation* {
                for (const auto& input : parent_output.get_target_inputs()) {
                    const auto child = ov::as_type<snippets::op::ConvertSaturation>(input.get_node());
                    if ((child != nullptr) && (child->get_output_element_type(0) == convert_type)) {
                        return child;
                    }
                }
                return nullptr;
            };

            for (size_t i = 0; i < op->get_input_size(); ++i) {
                const auto& op_input = op->input(i);
                const auto& required_after = precisions[i];
                auto parent_output = op_input.get_source_output();
                const auto actual_before = parent_output.get_element_type();
                if (actual_before != required_after) {
                    was_updated = true;
                    auto existing_convert = ov::as_type<ov::snippets::op::ConvertSaturation>(
                        parent_output.get_node());

                    if (existing_convert == nullptr) {
                        existing_convert = find_convert(parent_output, required_after);
                        if (existing_convert != nullptr) {
                            // reuse existing convert
                            op->set_argument(op_input.get_index(), existing_convert->shared_from_this());
                            continue;
                        }
                    }

                    if (existing_convert == nullptr) {
                        // create new Convert
                        auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(
                            parent_output,
                            required_after);
                        copy_runtime_info(parent_output.get_node_shared_ptr(), convert);
                        op->set_argument(op_input.get_index(), convert);
                        continue;
                    }

                    const auto actual_before = existing_convert->get_input_element_type(0);
                    const auto actual_after = existing_convert->get_output_element_type(0);

                    if (can_be_removed(actual_before, actual_after, required_after)) {
                        // remove existing convert
                        existing_convert->output(0).replace(parent_output);
                        continue;
                    }

                    if (can_be_fused(actual_after, required_after)) {
                        // fuse existing convert
                        auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(
                            existing_convert->get_input_node_shared_ptr(0),
                            required_after);
                        copy_runtime_info(parent_output.get_node_shared_ptr(), convert);
                        op->set_argument(op_input.get_index(), convert);
                        continue;
                    }

                    // create new convert
                    auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(
                        existing_convert->output(0),
                        required_after);
                    copy_runtime_info(existing_convert->output(0).get_node()->shared_from_this(), convert);
                    op->set_argument(op_input.get_index(), convert);
                }
            }
        }

        auto type_relaxed_node = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(op);
        if (was_updated || (type_relaxed_node != nullptr)) {
            const bool res = validate_and_infer_types_and_restore_outputs(op);
            was_updated = was_updated || res;
        }
    }

    for (auto it = result_types.begin(); it != result_types.end(); ++it) {
        const auto result = it->first;
        const auto actual_type = result->get_input_element_type(0);
        const auto expected_type = it->second;
        if (actual_type != it->second) {
            was_updated = true;
            auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(
                result->get_input_node_shared_ptr(0),
                expected_type);
            copy_runtime_info(result->get_input_node_shared_ptr(0), convert);
            result->set_argument(0, convert);
        }
    }

    return was_updated;
}

bool ov::snippets::pass::PropagatePrecision::validate_and_infer_types_and_restore_outputs(const std::shared_ptr<ov::Node>& op) {
    bool was_updated = false;

    // update output precision
    std::vector<element::Type> op_output_types;
    for (const auto& output : op->outputs()) {
        op_output_types.push_back(output.get_element_type());
    }

    auto type_relaxed_node = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(op);
    if (type_relaxed_node != nullptr) {
        // TODO: user story 104284
        // to keep previous functionality
        // unary and binary element-wise operations are supported
        // will be replaced to snippets opset later
        const auto op_element_type = op->get_input_element_type(0);
        if (type_relaxed_node->get_overridden_output_type(0) != op_element_type) {
            was_updated = true;
            OPENVINO_ASSERT(op->get_output_size() == 1ull, "operation with several output is not supported");

            type_relaxed_node->set_overridden_output_type(op_element_type, 0);
            op->validate_and_infer_types();
        }
    } else {
        op->validate_and_infer_types();
    }

    for (size_t i = 0; i < op->get_output_size(); ++i) {
        auto output = op->output(i);

        if (output.get_element_type() != op_output_types[i]) {
            was_updated = true;
            auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(
                output,
                op_output_types[i]);
            copy_runtime_info(output.get_node_shared_ptr(), convert);

            for (auto& input : output.get_target_inputs()) {
                auto child = input.get_node();
                if (child == convert.get()) {
                    continue;
                }

                input.replace_source_output(convert->output(0));


                if (ov::is_type<ov::op::v0::Result>(input.get_node())) {
                    // Result input tensor name was changed, the name has to be restored
                    // task #107826
                    input.get_tensor_ptr()->add_names(output.get_tensor_ptr()->get_names());

                    const std::string original_name = op->get_friendly_name();
                    op->set_friendly_name(original_name + "_original");
                    convert->set_friendly_name(original_name);
                }
            }
            output.get_tensor_ptr()->set_names({});
        }
    }

    return was_updated;
}

bool ov::snippets::pass::PropagatePrecision::can_be_removed(const element::Type& actual_before,
                                                            const element::Type& actual_after,
                                                            const element::Type& required_after) {
    if (actual_before != required_after) {
        return false;
    }

    return can_be_fused(actual_after, actual_before);
}

bool ov::snippets::pass::PropagatePrecision::can_be_fused(const element::Type& actual, const element::Type& required) {
    if (actual == required) {
        return true;
    }

    // custom conditions: between int & float precisions
    if (((actual == element::bf16) || (actual == element::f16) || (actual == element::f32)) &&
        ((required == element::u8) || (required == element::i8))) {
        return true;
    }

    if ((actual == element::f32) && ((required == element::u16) || (required == element::i16))) {
        return true;
    }

    // general conditions: any new added precision will support
    return
        (actual.is_signed() == required.is_signed()) &&
        (actual.is_real() == required.is_real()) &&
        (actual.bitwidth() > required.bitwidth());
}

std::vector<ov::element::Type> ov::snippets::pass::PropagatePrecision::get_precisions(
    const std::vector<element::Type>& input_precisions,
    const std::set<std::vector<element::Type>>& supported_precisions_pack) {
    bool was_found = false;
    for (const auto& supported_precisions : supported_precisions_pack) {
        for (size_t i = 0; i < supported_precisions.size(); ++i) {
            const auto& supported_precision = supported_precisions[i];
            const auto& input_precision = input_precisions[i];
            if ((supported_precision.is_real() != input_precision.is_real()) ||
                (input_precision.bitwidth() > supported_precision.bitwidth())) {
                was_found = false;
                break;
            }

            was_found = true;
        }
        if (was_found) {
            return supported_precisions;
        }
    }

    if (!supported_precisions_pack.empty()) {
        return *supported_precisions_pack.begin();
    }

    return {};
}
