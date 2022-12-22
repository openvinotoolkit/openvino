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

#ifdef CPU_DEBUG_CAPS_SNIPPETS
#include "ngraph/pass/visualize_tree.hpp"
#endif

ngraph::snippets::pass::PropagatePrecision::PropagatePrecision(
    const ov::element::Type supported_precision,
    const std::shared_ptr<const TargetMachine>& target_machine) : supported_precision(supported_precision), target_machine(target_machine) {
}

bool ngraph::snippets::pass::PropagatePrecision::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(PropagatePrecision);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::PropagatePrecision")

    const auto& ops = f->get_ordered_ops();
    for (const auto& op : f->get_ordered_ops()) {
        // TODO: remove when default supported precisions will be empty (not FP32)
        if (ngraph::is_type<opset1::Constant>(op)) {
            continue;
        }

        auto type_info = op->get_type_info();
        if (!target_machine->has(type_info)) {
            throw ov::Exception(
                "operation '" + std::string(type_info.version_id) + "::" + std::string(type_info.name) + 
                "' was not found in target machine");
        }

        std::vector<InferenceEngine::Precision> input_precisions;
        for (const auto& input : op->inputs()) {
            const auto input_precision = input.get_source_output().get_element_type();
            const auto input_precision_ie = InferenceEngine::details::convertPrecision(input_precision);
            input_precisions.push_back(input_precision_ie);
        }
        
        const auto supported_precisions = target_machine->get_supported_precisions(type_info);
        if (supported_precisions.empty()) {
            continue;
        }

        std::cout << "PropagatePrecision: " <<  op->get_friendly_name() << std::endl;
        assert(
            std::all_of(
                supported_precisions.begin(), 
                supported_precisions.end(),
                           [&input_precisions](const std::vector<InferenceEngine::Precision>& precisions) {
                               return precisions.size() == input_precisions.size();
                           }) && "input precisions count is not equal for supported precisions");

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
                    auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(parent_output, element_type);
                    parent_output.remove_target_input(input);

                    input.replace_source_output(convert->output(0));
                }
            }
            op->validate_and_infer_types();
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