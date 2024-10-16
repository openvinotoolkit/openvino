// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "move_brgemm_repacking_out.hpp"

#include "snippets/utils/utils.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/tpp/x64/op/modifiers.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

#include "cpu_shape.h"
#include "utils/general_utils.h"


namespace ov {
namespace intel_cpu {

using namespace snippets::lowered;


pass::MoveBrgemmRepackingOut::MoveBrgemmRepackingOut() {
    MATCHER_SCOPE(MoveBrgemmRepackingOut);
    auto m_param = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>();
    auto m_copy_b = ov::pass::pattern::wrap_type<BrgemmCopyB>({m_param});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::MoveBrgemmRepackingOut")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& copy_b_in = pattern_map.at(m_param);
        const auto& copy_b_out = pattern_map.at(m_copy_b);
        const auto copy_b_node = copy_b_out.get_node_shared_ptr();

        const auto& in_desc = PortDescriptorUtils::get_port_descriptor_ptr(copy_b_node->input(0));
        const auto& layout = in_desc->get_layout();
        // TODO:
        // 1. handle copyB with compensations
        // 2. handle non-planar layout
        if (!ov::snippets::utils::is_planar_layout(layout) || copy_b_node->get_output_size() != 1 ||
            transformation_callback(copy_b_node))
            return false;
        std::cout << "[ INFO ] MoveBrgemmRepackingOut is finished\n";
        return ov::replace_output_update_name(copy_b_out, copy_b_in);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_copy_b, matcher_name);
    register_matcher(m, callback);
}
} // namespace intel_cpu
} // namespace ov
