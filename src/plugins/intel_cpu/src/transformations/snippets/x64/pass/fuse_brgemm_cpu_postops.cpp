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
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "transformations/snippets/x64/pass/lowered/brgemm_cpu_blocking.hpp"

namespace ov::intel_cpu {

using namespace dnnl::impl;
using namespace snippets::op;
using namespace snippets::lowered;
using namespace ov::pass::pattern;
using PortDescriptorUtils = snippets::lowered::PortDescriptorUtils;

namespace {
ov::pass::pattern::op::Predicate brgemm_predicate(
    [](const Output<Node>& output) {
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(output.get_node_shared_ptr());
        OPENVINO_ASSERT(brgemm != nullptr, "BrgemmCPU node is expected");
        // Note: postops are not supported in case of blocking enabled
        // Ticket: 165567
        return has_static_rank()(output) && consumers_count(1)(output) &&
               !pass::BrgemmCPUBlocking::is_kn_blocking_supported(brgemm);
    },
    "brgemm_predicate");

ov::pass::pattern::op::Predicate scalar_predicate(
    [](const Output<Node>& output) {
        return type_matches(ov::element::f32)(output);
    },
    "scalar_predicate");

}  // namespace

pass::FuseBrgemmCPUPostops::FuseBrgemmCPUPostops(std::set<size_t>& brgemm_external_params_idces)
    : m_brgemm_external_params_idces(brgemm_external_params_idces) {
    add_matcher<FuseConvert>();
    add_matcher<FuseUnaryEltwise>();
    add_matcher<FuseScaleShift>();
    add_matcher<FuseClip>();
    add_matcher<FuseScalarEltwise>();
    add_matcher<FuseBinaryEltwise>(m_external_params);
}

bool pass::FuseBrgemmCPUPostops::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(FuseBrgemmCPUPostops);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseBrgemmCPUPostops")
    const auto res = GraphRewrite::run_on_model(m);
    for (const auto& param : m_external_params) {
        m_brgemm_external_params_idces.insert(m->get_parameter_index(param));
    }
    return res;
}

pass::FuseConvert::FuseConvert() {
    MATCHER_SCOPE(FuseConvert);

    auto m_brgemm = wrap_type<BrgemmCPU>(brgemm_predicate);
    auto m_convert = wrap_type<ConvertSaturation>({m_brgemm});

    auto callback = [=](Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseConvert")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto convert = pattern_map.at(m_convert).get_node_shared_ptr();
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());

        // Ticket 165567: In case of AMX, brgemm may require two kernels: main and tail.
        // Intermediate results must be stored in the output buffer for accumulation with tail kernel results.
        // Forcing an output precision with a smaller bit width for output buffer causes out-of-bounds memory writes
        // during intermediate results storage, so the convert fusion is skipped in this case.
        if (brgemm_utils::with_amx(brgemm->get_type())) {
            const auto& cur_out_precision = brgemm->get_output_element_type(0);
            const auto& new_out_precision = convert->get_output_element_type(0);
            if (cur_out_precision.bitwidth() > new_out_precision.bitwidth()) {
                return false;
            }
        }

        auto postops_config = brgemm->get_postops_config();
        postops_config.forced_output_type = convert->get_output_element_type(0);
        brgemm->set_postops_config(postops_config);
        brgemm->set_friendly_name(convert->get_friendly_name());
        ov::copy_runtime_info({brgemm, convert}, brgemm);
        ov::replace_node(convert, brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_convert, matcher_name);
    register_matcher(m, callback);
}

pass::FuseUnaryEltwise::FuseUnaryEltwise() {
    MATCHER_SCOPE(FuseUnaryEltwise);

    auto m_brgemm = wrap_type<BrgemmCPU>(brgemm_predicate);
    auto m_round = wrap_type<ov::op::v5::Round>({m_brgemm});
    auto m_relu = wrap_type<ov::op::v0::Relu>({m_brgemm});
    auto m_postop = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{m_round, m_relu});
    auto callback = [=](Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseUnaryEltwise")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto post_op = pattern_map.at(m_postop).get_node_shared_ptr();
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());

        auto postops_config = brgemm->get_postops_config();
        postops_config.forced_output_type = post_op->get_output_element_type(0);

        auto append_eltwise = [&postops_config, &post_op](alg_kind_t alg_kind) {
            OPENVINO_ASSERT(postops_config.post_ops.append_eltwise(1.f, alg_kind, 0.f, 0.f) == dnnl_success,
                            "Failed to append unary eltwise ",
                            post_op);
        };

        if (pattern_map.count(m_round)) {
            const auto round = ov::as_type_ptr<ov::op::v5::Round>(post_op);
            switch (round->get_mode()) {
            case ov::op::v5::Round::RoundMode::HALF_TO_EVEN:
                append_eltwise(alg_kind_t::dnnl_eltwise_round_half_to_even);
                break;
            case ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO:
                append_eltwise(alg_kind_t::dnnl_eltwise_round_half_away_from_zero);
                break;
            default:
                OPENVINO_THROW("Unsupported round mode");
            }
        } else if (pattern_map.count(m_relu)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_relu);
        } else {
            OPENVINO_THROW("Unsupported unary operation: ", post_op);
        }

        brgemm->set_postops_config(postops_config);
        brgemm->set_friendly_name(post_op->get_friendly_name());
        ov::copy_runtime_info({brgemm, post_op}, brgemm);
        ov::replace_node(post_op, brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_postop, matcher_name);
    register_matcher(m, callback);
}

pass::FuseScalarEltwise::FuseScalarEltwise() {
    MATCHER_SCOPE(FuseScalarEltwise);

    // These predicates are used to skip the transformation in cases where a more optimized transformation is available
    ov::pass::pattern::op::Predicate not_scale_shift_pattern(
        [](const Output<Node>& output) {
            if (!consumers_count(1)(output)) {
                return true;
            }
            const auto& consumer = output.get_target_inputs().begin()->get_node();
            if (ov::is_type<ov::op::v1::Add>(consumer) && (is_type<Scalar>(consumer->get_input_node_shared_ptr(0)) ||
                                                           is_type<Scalar>(consumer->get_input_node_shared_ptr(1)))) {
                return false;
            }
            return true;
        },
        "not_scale_shift_pattern");
    ov::pass::pattern::op::Predicate not_clip_pattern(
        [](const Output<Node>& output) {
            if (!consumers_count(1)(output)) {
                return true;
            }
            const auto& consumer = output.get_target_inputs().begin()->get_node();
            if (ov::is_type<ov::op::v1::Minimum>(consumer) &&
                (is_type<Scalar>(consumer->get_input_node_shared_ptr(0)) ||
                 is_type<Scalar>(consumer->get_input_node_shared_ptr(1)))) {
                return false;
            }
            return true;
        },
        "not_clip_pattern");

    auto m_brgemm = wrap_type<BrgemmCPU>(brgemm_predicate);
    auto m_scalar = wrap_type<Scalar>(scalar_predicate);
    auto m_mul = wrap_type<ov::op::v1::Multiply>({m_brgemm, m_scalar}, not_scale_shift_pattern);
    auto m_add = wrap_type<ov::op::v1::Add>({m_brgemm, m_scalar});
    auto m_sub = wrap_type<ov::op::v1::Subtract>({m_brgemm, m_scalar});
    auto m_max = wrap_type<ov::op::v1::Maximum>({m_brgemm, m_scalar}, not_clip_pattern);
    auto m_min = wrap_type<ov::op::v1::Minimum>({m_brgemm, m_scalar});
    auto m_postop = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{m_mul, m_add, m_sub, m_max, m_min});

    auto callback = [=](Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseScalarEltwise")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto post_op = pattern_map.at(m_postop).get_node_shared_ptr();
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());
        const auto scalar = ov::as_type_ptr<Scalar>(pattern_map.at(m_scalar).get_node_shared_ptr());
        const auto scalar_value = scalar->get_value<float>();

        auto postops_config = brgemm->get_postops_config();
        postops_config.forced_output_type = post_op->get_output_element_type(0);

        auto append_eltwise = [&postops_config, &post_op](alg_kind_t alg_kind, float alpha, float beta) {
            OPENVINO_ASSERT(postops_config.post_ops.append_eltwise(1.f, alg_kind, alpha, beta) == dnnl_success,
                            "Failed to append scalar eltwise ",
                            post_op,
                            " to brgemm postops. Alpha = ",
                            alpha,
                            " Beta = ",
                            beta);
        };

        if (pattern_map.count(m_mul)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_linear, scalar_value, 0.f);
        } else if (pattern_map.count(m_add)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_linear, 1.f, scalar_value);
        } else if (pattern_map.count(m_sub)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_linear, 1.f, -scalar_value);
        } else if (pattern_map.count(m_max)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_clip, scalar_value, std::numeric_limits<float>::max());
        } else if (pattern_map.count(m_min)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_clip, -std::numeric_limits<float>::max(), scalar_value);
        } else {
            OPENVINO_THROW("Unexpected postop: ", post_op);
        }
        brgemm->set_postops_config(postops_config);
        brgemm->set_friendly_name(post_op->get_friendly_name());
        ov::copy_runtime_info({brgemm, post_op}, brgemm);
        ov::replace_node(post_op, brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_postop, matcher_name);
    register_matcher(m, callback);
}

pass::FuseBinaryEltwise::FuseBinaryEltwise(std::set<std::shared_ptr<ov::op::v0::Parameter>>& external_params)
    : m_external_params(external_params) {
    MATCHER_SCOPE(FuseBinaryEltwise);

    ov::pass::pattern::op::Predicate binary_input_predicate(
        [](const Output<Node>& output) {
            return has_static_shape()(output) && type_matches(ov::element::f32)(output) && consumers_count(1)(output);
        },
        "binary_input_predicate");

    auto m_brgemm = wrap_type<BrgemmCPU>(brgemm_predicate);
    auto m_postop_input = wrap_type<ov::op::v0::Parameter>(binary_input_predicate);
    auto m_rank_norm = optional<RankNormalization>(m_postop_input);
    auto m_mul = wrap_type<ov::op::v1::Multiply>({m_brgemm, m_rank_norm});
    auto m_add = wrap_type<ov::op::v1::Add>({m_brgemm, m_rank_norm});
    auto m_sub = wrap_type<ov::op::v1::Subtract>({m_brgemm, m_rank_norm});
    auto m_max = wrap_type<ov::op::v1::Maximum>({m_brgemm, m_rank_norm});
    auto m_min = wrap_type<ov::op::v1::Minimum>({m_brgemm, m_rank_norm});
    auto m_postop = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{m_mul, m_add, m_sub, m_max, m_min});

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
        }

        auto append_binary = [&postops_config, &post_op, &memory_desc](alg_kind_t alg_kind) {
            OPENVINO_ASSERT(
                postops_config.post_ops.append_binary(alg_kind, memory_desc.getDnnlDesc().get()) == dnnl_success,
                "Failed to append binary eltwise ",
                post_op,
                " to brgemm postops.");
        };

        if (pattern_map.count(m_mul)) {
            append_binary(alg_kind_t::dnnl_binary_mul);
        } else if (pattern_map.count(m_add)) {
            append_binary(alg_kind_t::dnnl_binary_add);
        } else if (pattern_map.count(m_sub)) {
            append_binary(alg_kind_t::dnnl_binary_sub);
        } else if (pattern_map.count(m_max)) {
            append_binary(alg_kind_t::dnnl_binary_max);
        } else if (pattern_map.count(m_min)) {
            append_binary(alg_kind_t::dnnl_binary_min);
        } else {
            OPENVINO_THROW("Unexpected postop: ", post_op);
        }

        brgemm->set_postops_config(postops_config);
        brgemm->add_postop_input(pattern_map.count(m_rank_norm) ? pattern_map.at(m_rank_norm) : parameter_out);
        brgemm->set_friendly_name(post_op->get_friendly_name());
        ov::copy_runtime_info({brgemm, post_op}, brgemm);
        ov::replace_node(post_op, brgemm);

        // Note: binary postop's output and the corresponding matmul's input are marked as ignored
        // since they shouldn't be processed by the common lowering pipeline,
        // and will be handled by the brgemm kernel itself
        PortDescriptorUtils::set_address_reg_type(brgemm->inputs().back());
        PortDescriptorUtils::set_address_reg_type(parameter_out);
        if (pattern_map.count(m_rank_norm)) {
            const auto rank_norm = pattern_map.at(m_rank_norm).get_node_shared_ptr();
            PortDescriptorUtils::set_address_reg_type(rank_norm->input(0));
            PortDescriptorUtils::set_address_reg_type(rank_norm->output(0));
        }

        m_external_params.insert(ov::as_type_ptr<ov::op::v0::Parameter>(parameter_out.get_node_shared_ptr()));
        m_fused_postops_count++;
        return true;
    };

    auto m = std::make_shared<Matcher>(m_postop, matcher_name);
    register_matcher(m, callback);
}

pass::FuseScaleShift::FuseScaleShift() {
    MATCHER_SCOPE(FuseScaleShift);

    auto m_brgemm = wrap_type<BrgemmCPU>(brgemm_predicate);
    auto m_alpha = wrap_type<Scalar>(scalar_predicate);
    auto m_scale = wrap_type<ov::op::v1::Multiply>({m_brgemm, m_alpha}, consumers_count(1));
    auto m_beta = wrap_type<Scalar>(scalar_predicate);
    auto m_shift = wrap_type<ov::op::v1::Add>({m_scale, m_beta});

    auto callback = [=](Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseScaleShift")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto scale = pattern_map.at(m_scale).get_node_shared_ptr();
        const auto shift = pattern_map.at(m_shift).get_node_shared_ptr();
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());

        const auto alpha = ov::as_type_ptr<Scalar>(pattern_map.at(m_alpha).get_node_shared_ptr())->get_value<float>();
        const auto beta = ov::as_type_ptr<Scalar>(pattern_map.at(m_beta).get_node_shared_ptr())->get_value<float>();

        auto postops_config = brgemm->get_postops_config();
        postops_config.forced_output_type = shift->get_output_element_type(0);

        OPENVINO_ASSERT(
            postops_config.post_ops.append_eltwise(1.f, alg_kind_t::dnnl_eltwise_linear, alpha, beta) == dnnl_success,
            "Failed to append scale-shift eltwise to brgemm postops. Alpha = ",
            alpha,
            ", Beta = ",
            beta);

        brgemm->set_postops_config(postops_config);
        brgemm->set_friendly_name(shift->get_friendly_name());
        ov::copy_runtime_info({brgemm, scale, shift}, brgemm);
        ov::replace_node(shift, brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_shift, matcher_name);
    register_matcher(m, callback);
}

pass::FuseClip::FuseClip() {
    MATCHER_SCOPE(FuseClip);

    auto m_brgemm = wrap_type<BrgemmCPU>(brgemm_predicate);
    auto m_in_low = wrap_type<Scalar>(scalar_predicate);
    auto m_max = wrap_type<ov::op::v1::Maximum>({m_brgemm, m_in_low}, consumers_count(1));
    auto m_in_high = wrap_type<Scalar>(scalar_predicate);
    auto m_min = wrap_type<ov::op::v1::Minimum>({m_max, m_in_high});

    auto callback = [=](Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseClip")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto max_op = pattern_map.at(m_max).get_node_shared_ptr();
        const auto min_op = pattern_map.at(m_min).get_node_shared_ptr();
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());

        const auto clip_min =
            ov::as_type_ptr<Scalar>(pattern_map.at(m_in_low).get_node_shared_ptr())->get_value<float>();
        const auto clip_max =
            ov::as_type_ptr<Scalar>(pattern_map.at(m_in_high).get_node_shared_ptr())->get_value<float>();

        auto postops_config = brgemm->get_postops_config();
        postops_config.forced_output_type = min_op->get_output_element_type(0);

        OPENVINO_ASSERT(
            postops_config.post_ops.append_eltwise(1.f, alg_kind_t::dnnl_eltwise_clip, clip_min, clip_max) ==
                dnnl_success,
            "Failed to append clip eltwise to brgemm postops. in_low = ",
            clip_min,
            ", in_high = ",
            clip_max);

        brgemm->set_postops_config(postops_config);
        brgemm->set_friendly_name(min_op->get_friendly_name());
        ov::copy_runtime_info({brgemm, max_op, min_op}, brgemm);
        ov::replace_node(min_op, brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_min, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
