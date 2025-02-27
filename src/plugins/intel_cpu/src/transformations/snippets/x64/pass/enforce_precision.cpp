// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "enforce_precision.hpp"

#include <memory>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/rt_info.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "snippets/itt.hpp"
#include "snippets/pass/propagate_precision.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::intel_cpu::pass;

EnforcePrecision::EnforcePrecision(
    const ov::element::Type source,
    const ov::element::Type target,
    const std::function<std::set<std::vector<ov::element::Type>>(const std::shared_ptr<ov::Node>& op)>&
        get_supported_precisions)
    : source(source),
      target(target),
      get_supported_precisions(get_supported_precisions == nullptr ? get_supported_precisions_default
                                                                   : get_supported_precisions) {
    OPENVINO_ASSERT(source != target, "source and target precisions have to be different");
}

bool EnforcePrecision::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(EnforcePrecision);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::EnforcePrecision")

    bool was_updated = false;
    for (const auto& op : f->get_ordered_ops()) {
        ov::op::util::process_subgraph(*this, op);

        const auto& precisions = get_supported_precisions(op);

        if (precisions.empty()) {
            continue;
        }

        std::vector<element::Type> actual_precisions;
        for (const auto& input : op->inputs()) {
            actual_precisions.push_back(input.get_element_type());
        }

        // we should specify what supported_precisions_to_enforce from precisions set will be used to enforce
        // if result precisions are not supported then don't enforce
        auto op_is_appropriate = true;
        std::vector<element::Type> supported_precisions_to_enforce;
        for (const auto& supported_precisions : precisions) {
            if (supported_precisions.size() != actual_precisions.size()) {
                continue;
            }

            auto port_has_to_be_handled = false;
            for (auto index = 0ull; index < supported_precisions.size(); ++index) {
                if ((supported_precisions[index] == target) && (actual_precisions[index] == source)) {
                    // actual input precision has to be enforced: at least one port has to be handled
                    port_has_to_be_handled = true;
                } else if ((supported_precisions[index] != element::dynamic) &&
                           (supported_precisions[index] != actual_precisions[index])) {
                    // actual input precision is not enforced but not supported, operation has to be ignored
                    op_is_appropriate = false;
                    break;
                }
            }
            if (!op_is_appropriate) {
                break;
            }

            if (port_has_to_be_handled) {
                supported_precisions_to_enforce = supported_precisions;
                break;
            }
        }

        if (!op_is_appropriate || supported_precisions_to_enforce.empty()) {
            continue;
        }

        const auto insert_convert = [](const Output<Node>& parent_output,
                                       const std::shared_ptr<Node>& op,
                                       const size_t input_index,
                                       const element::Type& target) {
            auto convert = std::make_shared<snippets::op::ConvertSaturation>(parent_output, target);
            ov::copy_runtime_info(parent_output.get_node_shared_ptr(), convert);
            op->set_argument(input_index, convert);
        };

        for (auto index = 0ull; index < supported_precisions_to_enforce.size(); ++index) {
            if ((supported_precisions_to_enforce[index] == target) || (actual_precisions[index] == source)) {
                const auto op_parent =
                    ov::as_type_ptr<snippets::op::ConvertSaturation>(op->get_input_node_shared_ptr(index));
                if ((op_parent != nullptr) && (op_parent->get_input_element_type(0) == target) &&
                    // we can remove existing convertion only if precisions before and after are appropriate for removal
                    snippets::pass::PropagatePrecision::can_be_removed(op_parent->get_input_element_type(0),
                                                                       actual_precisions[index],
                                                                       supported_precisions_to_enforce[index])) {
                    // remove convert
                    op_parent->output(0).replace(op_parent->get_input_source_output(0));
                    was_updated = true;
                } else if (supported_precisions_to_enforce[index] != actual_precisions[index]) {
                    insert_convert(op->get_input_source_output(index), op, index, target);
                    was_updated = true;
                }
            }
        }

        auto type_relaxed_node = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(op);
        if (was_updated || (type_relaxed_node != nullptr)) {
            const bool res = snippets::pass::PropagatePrecision::validate_and_infer_types_and_restore_outputs(op);
            was_updated = was_updated || res;
        }
    }

    return was_updated;
}

std::set<std::vector<ov::element::Type>> EnforcePrecision::get_supported_precisions_default(
    const std::shared_ptr<ov::Node>& op) noexcept {
    std::set<std::vector<ov::element::Type>> types;
    if (ov::is_type<snippets::op::Brgemm>(op)) {
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx_fp16)) {
            types.insert({element::f16, element::f16});
        }
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16)) {
            types.insert({element::bf16, element::bf16});
        }
    }
    return types;
}
