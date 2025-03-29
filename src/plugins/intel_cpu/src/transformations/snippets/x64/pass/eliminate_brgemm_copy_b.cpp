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

pass::EliminateBrgemmCopyB::EliminateBrgemmCopyB() {
    MATCHER_SCOPE(EliminateBrgemmCopyB);
    auto m_param = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>();
    auto m_rank_norm = ov::pass::pattern::optional<ov::snippets::op::RankNormalization>(m_param);
    auto m_copy_b = ov::pass::pattern::wrap_type<BrgemmCopyB>({m_param});

    auto callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::EliminateBrgemmCopyB")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& copy_b_out = pattern_map.at(m_copy_b);
        const auto copy_b_node = ov::as_type_ptr<BrgemmCopyB>(copy_b_out.get_node_shared_ptr());
        OPENVINO_ASSERT(copy_b_node, "BrgemmCopyB node is null in EliminateBrgemmCopyB transformation");

        const auto& in_desc = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(copy_b_node->input(0));
        const auto& layout = in_desc->get_layout();

        // TODO [157340]: support external repacking for copyB with compensations
        if (brgemm_utils::with_compensations(copy_b_node->get_type()) || transformation_callback(copy_b_node)) {
            return false;
        }

        // If there is non-planar layout, we should insert reshape to support shape inference
        if (!ov::snippets::utils::is_planar_layout(layout)) {
            const auto& subtensor = in_desc->get_subtensor();
            const auto& reshape = std::make_shared<ov::snippets::op::Reorder>(copy_b_node->input_value(0), layout);
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(reshape->input(0), subtensor, layout);
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(reshape->output(0), subtensor);
            return ov::replace_node_update_name(copy_b_node, reshape);
        }

        // If there is no layout, we can just remove BrgemmCopyB from the subgraph
        return ov::replace_output_update_name(copy_b_out, copy_b_node->input_value(0));
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_copy_b, matcher_name);
    register_matcher(m, callback);
}
}  // namespace ov::intel_cpu
