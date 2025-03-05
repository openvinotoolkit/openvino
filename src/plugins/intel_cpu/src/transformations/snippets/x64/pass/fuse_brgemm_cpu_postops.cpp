// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_brgemm_cpu_postops.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

using namespace snippets::lowered;

pass::FuseBrgemmCPUPostops::FuseBrgemmCPUPostops() {
    MATCHER_SCOPE(FuseBrgemmCPUPostops);
    auto m_brgemm = ov::pass::pattern::wrap_type<BrgemmCPU>();
    auto m_convert = ov::pass::pattern::optional<ov::snippets::op::ConvertSaturation>(m_brgemm);

    auto m_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_postop = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_convert, m_constant});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseBrgemmCPUPostops")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto post_op = pattern_map.at(m_postop).get_node_shared_ptr();
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());

        // Note: due to specific handling of commutative ops in matcher,
        // 0's input is not always brgemm (even if it is 0's in the matcher)
        const auto op_after_brgemm = pattern_map.count(m_convert) ? pattern_map.at(m_convert).get_node_shared_ptr() : post_op;
        if (op_after_brgemm->get_input_node_shared_ptr(0).get() != brgemm.get()) {
            std::cout << "Post operation's input node: " << op_after_brgemm->get_input_node_shared_ptr(0)
                      << "\n is not BrgemmCPU: " << brgemm << ". Skipping fusion." << std::endl;
            return false;
        }

        // Note: currently, only post ops which don't change the shape are supported
        if (brgemm->get_output_partial_shape(0) != post_op->get_output_partial_shape(0)) {
            std::cout << "Output shape of BrgemmCPU and post operation do not match. Skipping fusion." << std::endl;
            return false;
        }

        // Log the addition of the post operation
        std::cout << "Adding post operation: " << post_op->get_friendly_name()
                  << " to BrgemmCPU: " << brgemm->get_friendly_name() << std::endl;
        brgemm->add_post_op(post_op);

        // Log the replacement output
        auto replacement_output = post_op->input_value(0);
        std::cout << "Initial replacement output set to input value of post operation: "
                  << replacement_output.get_node()->get_friendly_name() << std::endl;

        if (pattern_map.count(m_convert)) {
            const auto convert = pattern_map.at(m_convert).get_node_shared_ptr();
            std::cout << "Convert operation found: " << convert->get_friendly_name() << std::endl;

            OPENVINO_ASSERT(convert->get_output_element_type(0) == brgemm->get_output_element_type(0),
                            "Unexpected type for brgemm output conversion: ",
                            convert->get_output_element_type(0));

            replacement_output = convert->input_value(0);
            std::cout << "Replacement output updated to input value of convert operation: "
                      << replacement_output.get_node()->get_friendly_name() << std::endl;
        }

        // Log the output replacement
        std::cout << "Replacing output of post operation: " << post_op->get_friendly_name()
                  << " with: " << replacement_output.get_node()->get_friendly_name() << std::endl;
        return ov::replace_output_update_name(post_op->output(0), replacement_output);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_postop, matcher_name);
    register_matcher(m, callback);
}
}  // namespace ov::intel_cpu
