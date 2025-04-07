// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eliminate_brgemm_copy_b.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/op/reorder.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"

namespace ov::intel_cpu {

bool pass::EliminateBrgemmCopyB::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(EliminateBrgemmCopyB);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::EliminateBrgemmCopyB")

    auto m_param = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>();
    auto m_rank_norm = ov::pass::pattern::optional<ov::snippets::op::RankNormalization>(m_param);
    auto m_copy_b = ov::pass::pattern::wrap_type<BrgemmCopyB>({m_rank_norm});
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(m_copy_b);

    bool status = false;
    for (const auto& n : model->get_ordered_ops()) {
        if (!matcher->match(n)) {
            continue;
        }

        const auto& pattern_map = matcher->get_pattern_value_map();
        const auto& copy_b_out = pattern_map.at(m_copy_b);
        const auto copy_b_node = ov::as_type_ptr<BrgemmCopyB>(copy_b_out.get_node_shared_ptr());
        OPENVINO_ASSERT(copy_b_node, "BrgemmCopyB node is null in EliminateBrgemmCopyB transformation");

        const auto& in_desc = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(copy_b_node->input(0));
        const auto& layout = in_desc->get_layout();

        // TODO [157340]: support external repacking for copyB with compensations
        if (brgemm_utils::with_compensations(copy_b_node->get_type()) || transformation_callback(copy_b_node)) {
            return false;
        }

        const auto& param = ov::as_type_ptr<ov::op::v0::Parameter>(pattern_map.at(m_param).get_node_shared_ptr());
        const auto param_idx = static_cast<size_t>(model->get_parameter_index(param));
        OPENVINO_ASSERT(param_idx < model->get_parameters().size(),
                        "Parameter index is invalid in EliminateBrgemmCopyB transformation");
        // Update external repacking config for the further pipeline stages to mark this input as repacked
        auto& config = m_constant_inputs_idxs.count(param_idx) == 0 ? m_repacked_runtime_inputs_config
                                                                    : m_repacked_constant_inputs_config;
        config[param_idx] = RepackedInput();

        // If there is non-planar layout, we should insert reshape to support shape inference
        if (!ov::snippets::utils::is_planar_layout(layout)) {
            const auto& subtensor = in_desc->get_subtensor();
            const auto& reorder = std::make_shared<ov::snippets::op::Reorder>(copy_b_node->input_value(0), layout);
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(reorder->input(0), subtensor, layout);
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(reorder->output(0), subtensor);

            OPENVINO_ASSERT(ov::replace_node_update_name(copy_b_node, reorder),
                            "Failed to replace output in EliminateBrgemmCopyB transformation");
        } else {
            OPENVINO_ASSERT(ov::replace_output_update_name(copy_b_out, copy_b_node->input_value(0)),
                            "Failed to replace output in EliminateBrgemmCopyB transformation");
        }

        status = true;
    }

    return status;
}
}  // namespace ov::intel_cpu
