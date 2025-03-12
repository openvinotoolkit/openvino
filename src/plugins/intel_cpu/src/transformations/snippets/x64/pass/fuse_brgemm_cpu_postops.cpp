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
using PortDescriptorUtils = snippets::lowered::PortDescriptorUtils;

pass::FuseBrgemmCPUPostops::FuseBrgemmCPUPostops() {
    MATCHER_SCOPE(FuseBrgemmCPUPostops);
    using namespace ov::pass::pattern;
    auto m_brgemm = wrap_type<BrgemmCPU>();
    auto m_optional_convert = optional<ov::snippets::op::ConvertSaturation>(m_brgemm);

    auto postop_input_predicate = [](const Output<Node>& output) {
        return has_static_shape()(output) && type_matches(ov::element::f32)(output);
    };

    auto m_postop_value = wrap_type<ov::op::v0::Constant, ov::op::v0::Parameter>(postop_input_predicate);
    auto m_postop = wrap_type<ov::op::v1::Multiply, ov::op::v1::Add>({m_optional_convert, m_postop_value});

    auto callback = [=](Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseBrgemmCPUPostops")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto post_op = pattern_map.at(m_postop).get_node_shared_ptr();
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());

        // Note: due to specific handling of commutative ops in matcher,
        // 0's input is not always brgemm (even if it is 0's in the matcher)
        const auto op_after_brgemm =
            pattern_map.count(m_optional_convert) ? pattern_map.at(m_optional_convert).get_node_shared_ptr() : post_op;
        if (op_after_brgemm->get_input_node_shared_ptr(0).get() != brgemm.get()) {
            return false;
        }

        // Note: currently, only post ops which don't change the shape are supported
        if (brgemm->get_output_partial_shape(0) != post_op->get_output_partial_shape(0)) {
            return false;
        }

        // Log the addition of the post operation
        std::cout << "[ INFO ] Adding post operation: " << post_op->get_friendly_name()
                  << " to BrgemmCPU: " << brgemm->get_friendly_name() << std::endl;

        auto brgemm_inputs = brgemm->input_values();
        auto input_descs = brgemm->get_input_port_descriptors();
        for (size_t i = 1; i < post_op->get_input_size(); ++i) {
            brgemm_inputs.push_back(post_op->input_value(i));
            input_descs.push_back(ov::snippets::modifier::MemoryAccess::PortDescriptor{0, 0});
            const auto input_node = post_op->get_input_node_shared_ptr(i);
            input_node->get_rt_info()["POSTOP_INPUT"] = true;
        }

        auto postops = brgemm->get_postops();
        postops.push_back(post_op->get_type_info());

        auto new_brgemm = std::make_shared<BrgemmCPU>(
            brgemm_inputs,
            brgemm->get_type(),
            input_descs,
            // TODO: rewrite
            brgemm->get_output_port_descriptors().back(),
            PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(0))->get_layout(),
            PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(1))->get_layout(),
            PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0))->get_layout(),
            postops);
        new_brgemm->set_friendly_name(brgemm->get_friendly_name());
        ov::copy_runtime_info({brgemm, post_op}, new_brgemm);

        // PortDescriptors are copied manually since it is not copyable attribute
        for (size_t i = 0; i < brgemm->get_input_size(); ++i) {
            const auto in_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(i));
            PortDescriptorUtils::set_port_descriptor(new_brgemm->input(i), in_desc->get_subtensor(), in_desc->get_layout());
        }
        const auto out_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0));
        PortDescriptorUtils::set_port_descriptor(new_brgemm->output(0), out_desc->get_subtensor(), out_desc->get_layout());

        ov::replace_node(post_op, new_brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_postop, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
