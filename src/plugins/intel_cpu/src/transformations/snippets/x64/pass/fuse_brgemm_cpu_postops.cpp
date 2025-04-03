// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_brgemm_cpu_postops.hpp"

#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/rank_normalization.hpp"
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
std::shared_ptr<BrgemmCPU> clone_with_new_params(
    const std::shared_ptr<const BrgemmCPU>& brgemm,
    const BrgemmCPU::PostopsConfig& postops,
    const ov::OutputVector& new_inputs,
    const std::vector<ov::snippets::modifier::MemoryAccess::PortDescriptor>& new_in_descs) {
    auto new_brgemm =
        std::make_shared<BrgemmCPU>(new_inputs,
                                    brgemm->get_type(),
                                    new_in_descs,
                                    brgemm->get_output_port_descriptor(0),
                                    PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(0))->get_layout(),
                                    PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(1))->get_layout(),
                                    PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0))->get_layout(),
                                    postops);
    new_brgemm->set_friendly_name(brgemm->get_friendly_name());

    // PortDescriptors are copied manually since it is not copyable attribute
    for (size_t i = 0; i < brgemm->get_input_size(); ++i) {
        const auto in_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(i));
        PortDescriptorUtils::set_port_descriptor_ptr(new_brgemm->input(i), in_desc);
    }
    const auto out_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0));
    PortDescriptorUtils::set_port_descriptor_ptr(new_brgemm->output(0), out_desc);
    return new_brgemm;
}

auto common_brgemm_predicate = [](const Output<Node>& output) {
    return has_static_rank()(output) && consumers_count(1)(output);
};

}  // namespace

pass::FuseConvert::FuseConvert() {
    MATCHER_SCOPE(FuseConvert);

    auto m_brgemm = wrap_type<BrgemmCPU>(common_brgemm_predicate);
    auto m_convert = wrap_type<ov::snippets::op::ConvertSaturation>(
        {m_brgemm},
        type_matches_any({ov::element::f32, ov::element::i8, ov::element::u8}));

    auto callback = [=](Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseConvert")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto convert = pattern_map.at(m_convert).get_node_shared_ptr();
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());

        auto postops_config = brgemm->get_postops_config();
        postops_config.forced_output_type = convert->get_output_element_type(0);
        std::cout << "[ INFO ] FuseConvert fused convert with out precision: " << convert->get_output_element_type(0)
                  << std::endl;
        auto new_brgemm =
            clone_with_new_params(brgemm, postops_config, brgemm->input_values(), brgemm->get_input_port_descriptors());
        ov::copy_runtime_info({brgemm, convert}, new_brgemm);
        ov::replace_node(convert, new_brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_convert, matcher_name);
    register_matcher(m, callback);
}

pass::FuseScalarEltwise::FuseScalarEltwise() {
    MATCHER_SCOPE(FuseScalarEltwise);

    auto m_brgemm = wrap_type<BrgemmCPU>(common_brgemm_predicate);
    auto m_scalar = wrap_type<ov::snippets::op::Scalar>(type_matches(ov::element::f32));
    auto m_scale = wrap_type<ov::op::v1::Multiply>({m_brgemm, m_scalar});
    auto m_shift = wrap_type<ov::op::v1::Add>({m_brgemm, m_scalar});
    auto m_max = wrap_type<ov::op::v1::Maximum>({m_brgemm, m_scalar});
    auto m_min = wrap_type<ov::op::v1::Minimum>({m_brgemm, m_scalar});
    auto m_postop = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{m_scale, m_shift, m_max, m_min});

    auto callback = [=](Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseScalarEltwise")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto post_op = pattern_map.at(m_postop).get_node_shared_ptr();
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());
        const auto scalar = ov::as_type_ptr<ov::snippets::op::Scalar>(pattern_map.at(m_scalar).get_node_shared_ptr());
        const auto scalar_value = scalar->get_value<float>();

        auto postops_config = brgemm->get_postops_config();
        postops_config.forced_output_type = post_op->get_output_element_type(0);

        using namespace dnnl::impl;
        auto append_eltwise = [&postops_config, &post_op](alg_kind_t alg_kind, float alpha, float beta) {
            OPENVINO_ASSERT(postops_config.post_ops.append_eltwise(1.f, alg_kind, alpha, beta) == dnnl_success,
                            "Failed to append scalar eltwise ",
                            post_op,
                            " to brgemm postops. Alpha = ",
                            alpha,
                            " Beta = ",
                            beta);
        };

        if (pattern_map.count(m_scale)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_linear, scalar_value, 0.f);
            std::cout << "[ INFO ] FuseScalarEltwise fused scale: " << scalar_value << std::endl;
        } else if (pattern_map.count(m_shift)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_linear, 1.f, scalar_value);
            std::cout << "[ INFO ] FuseScalarEltwise fused shift: " << scalar_value << std::endl;
        } else if (pattern_map.count(m_max)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_clip, scalar_value, std::numeric_limits<float>::max());
            std::cout << "[ INFO ] FuseScalarEltwise fused max: " << scalar_value << std::endl;
        } else if (pattern_map.count(m_min)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_clip, -std::numeric_limits<float>::max(), scalar_value);
            std::cout << "[ INFO ] FuseScalarEltwise fused min: " << scalar_value << std::endl;
        } else {
            OPENVINO_THROW("Unexpected postop: ", post_op);
        }
        auto new_brgemm =
            clone_with_new_params(brgemm, postops_config, brgemm->input_values(), brgemm->get_input_port_descriptors());
        ov::copy_runtime_info({brgemm, post_op}, new_brgemm);
        ov::replace_node(post_op, new_brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_postop, matcher_name);
    register_matcher(m, callback);
}

pass::FuseBinaryEltwise::FuseBinaryEltwise(std::set<std::shared_ptr<ov::op::v0::Parameter>>& external_params)
    : m_external_params(external_params) {
    MATCHER_SCOPE(FuseBinaryEltwise);

    auto binary_input_predicate = [](const Output<Node>& output) {
        return has_static_shape()(output) && type_matches(ov::element::f32)(output) && consumers_count(1)(output);
    };

    auto brgemm_predicate = [](const Output<Node>& output) {
        const auto brgemm = output.get_node_shared_ptr();
        // Note: binary postops are not supported in case of blocking enabled,
        // so f32 precision is not included in supported list
        static const ov::element::TypeVector supported_in_precisions{ov::element::bf16,
                                                                     ov::element::i8,
                                                                     ov::element::u8};
        return common_brgemm_predicate(output) &&
               type_matches_any(supported_in_precisions)(brgemm->input_value(0)) &&
               type_matches_any(supported_in_precisions)(brgemm->input_value(1));
    };

    auto m_brgemm = wrap_type<BrgemmCPU>(brgemm_predicate);
    auto m_postop_input = wrap_type<ov::op::v0::Parameter>(binary_input_predicate);
    auto m_rank_norm = optional<ov::snippets::op::RankNormalization>(m_postop_input);
    auto m_mul = wrap_type<ov::op::v1::Multiply>({m_brgemm, m_rank_norm});
    auto m_add = wrap_type<ov::op::v1::Add>({m_brgemm, m_rank_norm});
    auto m_max = wrap_type<ov::op::v1::Maximum>({m_brgemm, m_rank_norm});
    auto m_min = wrap_type<ov::op::v1::Minimum>({m_brgemm, m_rank_norm});
    auto m_postop = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{m_mul, m_add, m_max, m_min});

    auto callback = [=](Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseScalarEltwise")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto post_op = pattern_map.at(m_postop).get_node_shared_ptr();
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());
        const auto& output_shape = brgemm->get_output_partial_shape(0);
        const auto OC_dim = *output_shape.rbegin();
        if (OC_dim.is_dynamic()) {
            return false;
        }
        const size_t OC = OC_dim.get_length();

        const auto& parameter_out = pattern_map.at(m_postop_input);
        const auto& parameter_shape = parameter_out.get_shape();
        if (ov::shape_size(parameter_shape) != OC || parameter_shape.back() != OC) {
            return false;
        }

        VectorDims per_channel_shape = {1, OC};
        DnnlBlockedMemoryDesc memory_desc(ov::element::f32, Shape(per_channel_shape));

        auto postops_config = brgemm->get_postops_config();
        postops_config.forced_output_type = post_op->get_output_element_type(0);
        if (!postops_config.binary_postops_offset) {
            postops_config.binary_postops_offset = m_fused_postops_count;
            std::cout << "[ INFO ] binary_postops_offset is set to " << m_fused_postops_count << std::endl;
        } else {
            std::cout << "[ INFO ] binary postops is already set to " << postops_config.binary_postops_offset.value()
                  << std::endl;
        }

        using namespace dnnl::impl;
        auto append_binary = [&postops_config, &post_op, &memory_desc](alg_kind_t alg_kind) {
            OPENVINO_ASSERT(
                postops_config.post_ops.append_binary(alg_kind, memory_desc.getDnnlDesc().get()) == dnnl_success,
                "Failed to append binary eltwise ",
                post_op,
                " to brgemm postops.");
        };

        if (pattern_map.count(m_mul)) {
            append_binary(alg_kind_t::dnnl_binary_mul);
            std::cout << "[ INFO ] FuseBinaryEltwise fused binary mul\n";
        } else if (pattern_map.count(m_add)) {
            append_binary(alg_kind_t::dnnl_binary_add);
            std::cout << "[ INFO ] FuseBinaryEltwise fused binary add\n";
        } else if (pattern_map.count(m_max)) {
            append_binary(alg_kind_t::dnnl_binary_max);
            std::cout << "[ INFO ] FuseBinaryEltwise fused binary max\n";
        } else if (pattern_map.count(m_min)) {
            append_binary(alg_kind_t::dnnl_binary_min);
            std::cout << "[ INFO ] FuseBinaryEltwise fused binary min\n";
        } else {
            OPENVINO_THROW("Unexpected postop: ", post_op);
        }

        const auto postop_input_node = ov::as_type_ptr<ov::op::v0::Parameter>(parameter_out.get_node_shared_ptr());
        OPENVINO_ASSERT(postop_input_node != nullptr,
                        "postop_input_node is nullptr. It should be a Parameter node");
        m_external_params.insert(postop_input_node);

        auto brgemm_inputs = brgemm->input_values();
        auto input_descs = brgemm->get_input_port_descriptors();
        brgemm_inputs.push_back(pattern_map.count(m_rank_norm) ? pattern_map.at(m_rank_norm) : parameter_out);
        input_descs.push_back(ov::snippets::modifier::MemoryAccess::PortDescriptor{0, 0});

        auto new_brgemm = clone_with_new_params(brgemm, postops_config, brgemm_inputs, input_descs);
        ov::copy_runtime_info({brgemm, post_op}, new_brgemm);
        ov::replace_node(post_op, new_brgemm);

        // Note: binary postop's output and the corresponding matmul's input are marked as ignored
        // since they shouldn't be processed by the common lowering pipeline,
        // and will be handled by the brgemm kernel itself
        PortDescriptorUtils::set_ignored_reg_type(new_brgemm->inputs().back());
        PortDescriptorUtils::set_ignored_reg_type(parameter_out);
        if (pattern_map.count(m_rank_norm)) {
            const auto rank_norm = pattern_map.at(m_rank_norm).get_node_shared_ptr();
            PortDescriptorUtils::set_ignored_reg_type(rank_norm->input(0));
            PortDescriptorUtils::set_ignored_reg_type(rank_norm->output(0));
        }
        m_fused_postops_count++;
        return true;
    };

    auto m = std::make_shared<Matcher>(m_postop, matcher_name);
    register_matcher(m, callback);
}

bool pass::FuseBrgemmCPUPostops::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(FuseBrgemmCPUPostops);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseBrgemmCPUPostops")
    const auto res = GraphRewrite::run_on_model(m);
    for (const auto& param : m_external_params) {
        std::cout << " [ INFO ] FuseBrgemmCPUPostops::run_on_model: adding param with index "
                  << m->get_parameter_index(param) << std::endl;
        m_brgemm_external_params_idces.insert(m->get_parameter_index(param));
    }
    return res;
}

}  // namespace ov::intel_cpu
