// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_brgemm_repacking_out.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"

namespace ov {
namespace intel_cpu {

pass::MoveBrgemmRepackingOut::MoveBrgemmRepackingOut() {
    MATCHER_SCOPE(MoveBrgemmRepackingOut);
    auto m_param = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>();
    auto m_rank_norm = ov::pass::pattern::optional<ov::snippets::op::RankNormalization>(m_param);
    auto m_copy_b = ov::pass::pattern::wrap_type<BrgemmCopyB>({m_param});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::MoveBrgemmRepackingOut")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& copy_b_out = pattern_map.at(m_copy_b);
        const auto copy_b_node = ov::as_type_ptr<BrgemmCopyB>(copy_b_out.get_node_shared_ptr());
        OPENVINO_ASSERT(copy_b_node, "BrgemmCopyB node is null in MoveBrgemmRepackingOut transformation");

        const auto& in_desc = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(copy_b_node->input(0));
        const auto& layout = in_desc->get_layout();
        // TODO:
        // 1. Ticket 157340: support external repacking for copyB with compensations
        // 2. Ticket 157339: support external repacking for non-planar layout
        if (!ov::snippets::utils::is_planar_layout(layout) ||
            copy_b_node->get_src_element_type() == ov::element::i8 || transformation_callback(copy_b_node))
            return false;
        return ov::replace_output_update_name(copy_b_out, copy_b_node->input_value(0));
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_copy_b, matcher_name);
    register_matcher(m, callback);
}
} // namespace intel_cpu
} // namespace ov
