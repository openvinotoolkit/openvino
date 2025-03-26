// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_brgemm_cpu_postops.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

using namespace snippets::lowered;
using namespace ov::pass::pattern;
using PortDescriptorUtils = snippets::lowered::PortDescriptorUtils;

namespace {
auto brgemm_predicate = [](const Output<Node>& output) {
    return has_static_shape()(output) && consumers_count(1)(output);
};

auto scalar_predicate = [](const Output<Node>& output) {
    return has_static_shape()(output) && type_matches(ov::element::f32)(output) &&
           ov::shape_size(output.get_shape()) == 1;
};

std::shared_ptr<BrgemmCPU> clone_with_new_params(
    const std::shared_ptr<BrgemmCPU>& brgemm,
    const BrgemmCPU::PostopsConfig& postops,
    const ov::OutputVector& new_inputs,
    const std::vector<ov::snippets::modifier::MemoryAccess::PortDescriptor>& new_in_descs) {
    auto new_brgemm =
        std::make_shared<BrgemmCPU>(new_inputs,
                                    brgemm->get_type(),
                                    new_in_descs,
                                    // TODO: rewrite
                                    brgemm->get_output_port_descriptors().back(),
                                    PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(0))->get_layout(),
                                    PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(1))->get_layout(),
                                    PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0))->get_layout(),
                                    postops);
    new_brgemm->set_friendly_name(brgemm->get_friendly_name());

    // PortDescriptors are copied manually since it is not copyable attribute
    for (size_t i = 0; i < brgemm->get_input_size(); ++i) {
        const auto in_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(i));
        PortDescriptorUtils::set_port_descriptor(new_brgemm->input(i), in_desc->get_subtensor(), in_desc->get_layout());
    }
    const auto out_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0));
    PortDescriptorUtils::set_port_descriptor(new_brgemm->output(0), out_desc->get_subtensor(), out_desc->get_layout());
    return new_brgemm;
}

} // namespace

pass::FuseScaleShift::FuseScaleShift() {
    MATCHER_SCOPE(FuseScaleShift);

    auto m_brgemm = wrap_type<BrgemmCPU>(brgemm_predicate);
    auto m_optional_convert = optional<ov::snippets::op::ConvertSaturation>(m_brgemm);

    auto m_scalar = wrap_type<ov::snippets::op::Scalar>();
    auto m_scale = wrap_type<ov::op::v1::Multiply>({m_optional_convert, m_scalar});
    auto m_shift = wrap_type<ov::op::v1::Add>({m_optional_convert, m_scalar});
    auto m_postop = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{m_scale, m_shift});

    auto callback = [=](Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseScaleShift")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto post_op = pattern_map.at(m_postop).get_node_shared_ptr();
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());
        const auto scalar = ov::as_type_ptr<ov::snippets::op::Scalar>(pattern_map.at(m_scalar).get_node_shared_ptr());
        const auto scalar_value = scalar->get_value<float>();

        auto postops = brgemm->get_postops();
        if (pattern_map.count(m_scale)) {
            OPENVINO_ASSERT(postops.append_eltwise(1.f,
                                                   dnnl::impl::alg_kind_t::dnnl_eltwise_linear,
                                                   scalar_value,
                                                   0.f) == dnnl_success);
            std::cout << "[ INFO ] FuseScaleShift fused scale: " << scalar_value << std::endl;
        } else if (pattern_map.count(m_shift)) {
            OPENVINO_ASSERT(postops.append_eltwise(1.f,
                                                   dnnl::impl::alg_kind_t::dnnl_eltwise_linear,
                                                   1.f,
                                                   scalar_value) == dnnl_success);
            std::cout << "[ INFO ] FuseScaleShift fused shift: " << scalar_value << std::endl;
        }
        auto new_brgemm =
            clone_with_new_params(brgemm, postops, brgemm->input_values(), brgemm->get_input_port_descriptors());
        ov::copy_runtime_info({brgemm, post_op}, new_brgemm);
        ov::replace_node(post_op, new_brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_postop, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
