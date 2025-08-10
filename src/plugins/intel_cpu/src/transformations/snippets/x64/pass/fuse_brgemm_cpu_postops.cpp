// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_brgemm_cpu_postops.hpp"

#include <common/c_types_map.hpp>
#include <cstddef>
#include <limits>
#include <memory>
#include <set>

#include "cpu_shape.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/predicate.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/pass/lowered/brgemm_cpu_blocking.hpp"

namespace ov::intel_cpu {

using namespace dnnl::impl;
using namespace snippets::op;
using namespace snippets::lowered;
using namespace ov::pass::pattern;
using PortDescriptorUtils = snippets::lowered::PortDescriptorUtils;

bool pass::FuseBrgemmCPUPostops::brgemm_can_fuse_postop(const ov::element::Type& input_precision) {
    // Note: postops are not supported in case of blocking enabled
    // Ticket: 165567
    return !pass::BrgemmCPUBlocking::is_kn_blocking_supported(input_precision);
}

namespace {
const ov::pass::pattern::op::Predicate brgemm_predicate(
    [](const Output<Node>& output) {
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(output.get_node_shared_ptr());
        return has_static_rank()(output) && consumers_count(1)(output) && brgemm != nullptr &&
               pass::FuseBrgemmCPUPostops::brgemm_can_fuse_postop(brgemm->get_input_element_type(1));
    },
    "brgemm_predicate");

const ov::pass::pattern::op::Predicate scalar_predicate(
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

bool pass::FuseBrgemmCPUPostops::can_be_fused_as_postop(const std::shared_ptr<const ov::Node>& node) {
    return pass::FuseUnaryEltwise::can_be_fused(node) || pass::FuseScalarEltwise::can_be_fused(node) ||
           pass::FuseBinaryEltwise::can_be_fused(node);
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
        // during intermediate results storage, so the convert fusion is skipped in the case when internal blocking is
        // needed.
        if (brgemm->get_config().is_amx()) {
            const auto& cur_out_precision = brgemm->get_output_element_type(0);
            const auto& new_out_precision = convert->get_output_element_type(0);
            if (cur_out_precision.bitwidth() > new_out_precision.bitwidth()) {
                const auto a_shape = ov::snippets::utils::get_planar_pshape(brgemm->input(0));
                const auto& k_dim = *a_shape.rbegin();
                const auto k_inner_block = brgemm->get_config().wei_k_blk();
                if (k_dim.is_dynamic() || (k_dim.get_length() % k_inner_block != 0)) {
                    return false;
                }
            }
        }

        brgemm->force_output_type(convert->get_output_element_type(0));
        brgemm->set_friendly_name(convert->get_friendly_name());
        ov::copy_runtime_info({brgemm, convert}, brgemm);
        ov::replace_node(convert, brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_convert, matcher_name);
    register_matcher(m, callback);
}

bool pass::FuseUnaryEltwise::can_be_fused(const std::shared_ptr<const ov::Node>& node) {
    return ov::is_type_any_of<ov::op::v5::Round, ov::op::v0::Relu>(node);
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
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());
        OPENVINO_ASSERT(brgemm, "BrgemmCPU node is expected");
        const auto post_op = pattern_map.at(m_postop).get_node_shared_ptr();
        if (!can_be_fused(post_op)) {
            return false;
        }

        auto append_eltwise = [&brgemm](alg_kind_t alg_kind) {
            brgemm->add_scalar_eltwise_postop(alg_kind, 0.F, 0.F);
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

        brgemm->set_friendly_name(post_op->get_friendly_name());
        ov::copy_runtime_info({brgemm, post_op}, brgemm);
        ov::replace_node(post_op, brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_postop, matcher_name);
    register_matcher(m, callback);
}

bool pass::FuseScalarEltwise::can_be_fused(const std::shared_ptr<const ov::Node>& node) {
    return ov::is_type_any_of<ov::op::v1::Multiply,
                              ov::op::v1::Add,
                              ov::op::v1::Subtract,
                              ov::op::v1::Maximum,
                              ov::op::v1::Minimum>(node) &&
           node->get_input_partial_shape(1).is_static() && ov::shape_size(node->get_input_shape(1)) == 1;
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
            const bool is_scale_shift_pattern =
                ov::is_type<ov::op::v1::Add>(consumer) && (is_type<Scalar>(consumer->get_input_node_shared_ptr(0)) ||
                                                           is_type<Scalar>(consumer->get_input_node_shared_ptr(1)));
            return !is_scale_shift_pattern;
        },
        "not_scale_shift_pattern");
    ov::pass::pattern::op::Predicate not_clip_pattern(
        [](const Output<Node>& output) {
            if (!consumers_count(1)(output)) {
                return true;
            }
            const auto& consumer = output.get_target_inputs().begin()->get_node();
            const bool is_clip_pattern = ov::is_type<ov::op::v1::Minimum>(consumer) &&
                                         (is_type<Scalar>(consumer->get_input_node_shared_ptr(0)) ||
                                          is_type<Scalar>(consumer->get_input_node_shared_ptr(1)));
            return !is_clip_pattern;
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
        if (!can_be_fused(post_op)) {
            return false;
        }

        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());
        const auto scalar = ov::as_type_ptr<Scalar>(pattern_map.at(m_scalar).get_node_shared_ptr());
        OPENVINO_ASSERT(brgemm != nullptr && scalar != nullptr, "BrgemmCPU node and scalar are expected");
        const auto scalar_value = scalar->get_value<float>();

        auto append_eltwise = [&brgemm](alg_kind_t alg_kind, float alpha, float beta) {
            brgemm->add_scalar_eltwise_postop(alg_kind, alpha, beta);
        };

        if (pattern_map.count(m_mul)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_linear, scalar_value, 0.F);
        } else if (pattern_map.count(m_add)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_linear, 1.F, scalar_value);
        } else if (pattern_map.count(m_sub)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_linear, 1.F, -scalar_value);
        } else if (pattern_map.count(m_max)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_clip, scalar_value, std::numeric_limits<float>::max());
        } else if (pattern_map.count(m_min)) {
            append_eltwise(alg_kind_t::dnnl_eltwise_clip, -std::numeric_limits<float>::max(), scalar_value);
        } else {
            OPENVINO_THROW("Unexpected postop: ", post_op);
        }
        brgemm->set_friendly_name(post_op->get_friendly_name());
        ov::copy_runtime_info({brgemm, post_op}, brgemm);
        ov::replace_node(post_op, brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_postop, matcher_name);
    register_matcher(m, callback);
}

ov::pass::pattern::op::Predicate pass::FuseBinaryEltwise::binary_input_predicate{
    [](const Output<Node>& output) {
        return has_static_shape()(output) && type_matches(ov::element::f32)(output) && consumers_count(1)(output);
    },
    "binary_input_predicate"};

bool pass::FuseBinaryEltwise::can_be_fused(const std::shared_ptr<const ov::Node>& node) {
    if (!ov::is_type_any_of<ov::op::v1::Multiply,
                            ov::op::v1::Add,
                            ov::op::v1::Subtract,
                            ov::op::v1::Maximum,
                            ov::op::v1::Minimum>(node)) {
        return false;
    }
    const auto& output_shape = node->get_output_partial_shape(0);
    if (output_shape.rank().is_dynamic() || output_shape.rbegin()->is_dynamic()) {
        return false;
    }
    const size_t OC = output_shape.rbegin()->get_length();
    const auto& postop_input = node->input_value(1);
    return binary_input_predicate(postop_input) && ov::shape_size(postop_input.get_shape()) == OC &&
           postop_input.get_shape().back() == OC;
}

pass::FuseBinaryEltwise::FuseBinaryEltwise(std::set<std::shared_ptr<ov::op::v0::Parameter>>& external_params)
    : m_external_params(external_params) {
    MATCHER_SCOPE(FuseBinaryEltwise);

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
        if (!can_be_fused(post_op)) {
            return false;
        }

        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(pattern_map.at(m_brgemm).get_node_shared_ptr());
        OPENVINO_ASSERT(brgemm, "BrgemmCPU node is expected");

        const size_t OC = brgemm->get_output_partial_shape(0).rbegin()->get_length();
        const DnnlBlockedMemoryDesc memory_desc(ov::element::f32, Shape({1, OC}));
        const auto& parameter_out = pattern_map.at(m_postop_input);
        const auto& postop_input = pattern_map.count(m_rank_norm) ? pattern_map.at(m_rank_norm) : parameter_out;

        auto append_binary = [&](alg_kind_t alg_kind) {
            brgemm->add_binary_eltwise_postop(alg_kind, memory_desc.getDnnlDesc(), postop_input, m_fused_postops_count);
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

        brgemm->add_scalar_eltwise_postop(alg_kind_t::dnnl_eltwise_linear, alpha, beta);
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

        brgemm->add_scalar_eltwise_postop(alg_kind_t::dnnl_eltwise_clip, clip_min, clip_max);
        brgemm->set_friendly_name(min_op->get_friendly_name());
        ov::copy_runtime_info({brgemm, max_op, min_op}, brgemm);
        ov::replace_node(min_op, brgemm);
        return true;
    };

    auto m = std::make_shared<Matcher>(m_min, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
