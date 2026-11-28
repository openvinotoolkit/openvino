// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "l2_norm_to_rms.hpp"

#include <cmath>
#include <limits>
#include <optional>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/util/reduction_base.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
using ov::pass::pattern::op::Predicate;

std::optional<double> max_finite(const ov::element::Type& type) {
    if (type == ov::element::f16) {
        return static_cast<double>(static_cast<float>(std::numeric_limits<ov::float16>::max()));
    }
    if (type == ov::element::bf16) {
        return static_cast<double>(static_cast<float>(std::numeric_limits<ov::bfloat16>::max()));
    }
    if (type == ov::element::f32) {
        return static_cast<double>(std::numeric_limits<float>::max());
    }
    return std::nullopt;
}

std::optional<double> denorm_min(const ov::element::Type& type) {
    if (type == ov::element::f16) {
        return std::ldexp(1.0, -24);
    }
    if (type == ov::element::bf16) {
        return std::ldexp(1.0, -133);
    }
    if (type == ov::element::f32) {
        return static_cast<double>(std::numeric_limits<float>::denorm_min());
    }
    return std::nullopt;
}

std::optional<double> max_negligible_eps(const ov::element::Type& type) {
    const auto value = denorm_min(type);
    return value ? std::optional<double>(*value / 1024.0) : std::nullopt;
}

std::optional<double> scalar_value(const ov::Output<ov::Node>& out) {
    const ov::Node* node = out.get_node();
    if (ov::is_type<v0::Convert>(node)) {
        node = node->get_input_node_ptr(0);
    }
    const auto* const constant = ov::as_type<const v0::Constant>(node);
    if (!constant || ov::shape_size(constant->get_shape()) != 1) {
        return std::nullopt;
    }
    return constant->cast_vector<double>()[0];
}

// The custom predicates below check constant values against bounds or relate the shapes of an op's inputs; the
// built-in ones (value_matches, shape_matches, ...) compare against fixed literals only.

Predicate reduces_single_axis(int64_t axis) {
    return Predicate(
        [axis](const ov::Output<ov::Node>& out) {
            const auto* const reduce = ov::as_type<const ov::op::util::ReductionBase>(out.get_node());
            if (!reduce) {
                return false;
            }
            const auto rank = reduce->get_input_partial_shape(0).rank();
            if (rank.is_dynamic()) {
                return false;
            }
            const int64_t expected = axis < 0 ? rank.get_length() + axis : axis;
            return expected >= 0 && reduce->get_reduction_axes() == ov::AxisSet{static_cast<size_t>(expected)};
        },
        "reduces_single_axis(" + std::to_string(axis) + ")");
}

bool is_negligible_eps(double eps, const ov::element::Type& type) {
    const auto limit = max_negligible_eps(type);
    return limit && eps >= 0.0 && eps <= *limit;
}

Predicate is_negligible_eps_max() {
    return Predicate(
        [](const ov::Output<ov::Node>& out) {
            const auto value = scalar_value(out.get_node()->input_value(1));
            return value && is_negligible_eps(*value, out.get_element_type());
        },
        "is_negligible_eps_max");
}

Predicate is_no_op_clip() {
    return Predicate(
        [](const ov::Output<ov::Node>& out) {
            const auto value = scalar_value(out);
            const auto limit = max_finite(out.get_element_type());
            return value && limit && *value >= *limit;
        },
        "is_no_op_clip");
}

Predicate is_eps_clamp() {
    return Predicate(
        [](const ov::Output<ov::Node>& out) {
            const auto* const clamp = ov::as_type<const v0::Clamp>(out.get_node());
            const auto limit = max_finite(out.get_element_type());
            return clamp != nullptr && limit && is_negligible_eps(clamp->get_min(), out.get_element_type()) && clamp->get_max() >= *limit;
        },
        "is_eps_clamp");
}

// Constant whose elements are all equal; with `zero`, all equal to 0. value_matches("0") matches one element only,
// while a per-channel zero bias has C of them. Values outside the finite f32 range (e.g. inf) do not match, the same
// check the LastAxis callback asserts on.
Predicate is_uniform_constant(bool zero) {
    return Predicate(
        [zero](const ov::Output<ov::Node>& out) {
            float value = 0.F;
            const auto constant = ov::as_type_ptr<v0::Constant>(out.get_node_shared_ptr());
            return constant && ov::op::util::get_single_value(constant, value) && (!zero || value == 0.F);
        },
        zero ? "is_all_zero" : "is_uniform_constant");
}

// A static shape that, numpy-broadcast against a rank-`rank` tensor with `channels` channels on axis 1, is either a
// scalar or a vector along axis 1 (e.g. {C, 1, 1} or {1, C, 1, 1} for rank 4).
bool is_per_channel(const ov::PartialShape& pshape, size_t rank, size_t channels) {
    if (pshape.is_dynamic()) {
        return false;
    }
    const auto shape = pshape.to_shape();
    if (shape.size() > rank) {
        return false;
    }
    if (ov::shape_size(shape) == 1) {
        return true;
    }
    if (shape.size() + 1 < rank) {
        return false;
    }
    const size_t offset = rank - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != (i + offset == 1 ? channels : 1)) {
            return false;
        }
    }
    return true;
}

// Binary eltwise op on a rank-4/5 tensor that applies a scalar or per-channel operand without changing the shape of
// the other operand.
Predicate applies_per_channel() {
    return Predicate(
        [](const ov::Output<ov::Node>& out) {
            const auto& shape = out.get_partial_shape();
            if (shape.rank().is_dynamic() || (shape.size() != 4 && shape.size() != 5) || shape[1].is_dynamic()) {
                return false;
            }
            const auto* node = out.get_node();
            const auto channels = static_cast<size_t>(shape[1].get_length());
            for (size_t i = 0; i < 2; ++i) {
                if (node->get_input_partial_shape(i) == shape && is_per_channel(node->get_input_partial_shape(1 - i), shape.size(), channels)) {
                    return true;
                }
            }
            return false;
        },
        "applies_per_channel");
}

}  // namespace

L2NormToRMSLastAxis::L2NormToRMSLastAxis(size_t max_work_group_size) {
    using namespace ov::pass::pattern;

    // The GPU RMS kernel has bfyx and bfzyx implementations only, so rank <= 5; N goes into the rewritten gamma and eps,
    // so it must be static; and one work group handles a row (the limit RMSFusion applies on GPU, 8 elements per item).
    const auto fits_rms_kernel = Predicate(
        [max_work_group_size](const ov::Output<ov::Node>& out) {
            const auto& shape = out.get_node()->get_input_partial_shape(0);
            if (shape.rank().is_dynamic() || shape.size() == 0 || shape.size() > 5 || shape[shape.size() - 1].is_dynamic()) {
                return false;
            }
            const size_t vec_size = 8;
            return static_cast<size_t>(shape[shape.size() - 1].get_length()) / vec_size <= max_work_group_size;
        },
        "fits_rms_kernel");
    // gamma {..., 1, N}: N elements along the normalized axis. A gamma with N elements along another axis, e.g. {N, 1},
    // broadcasts per row instead.
    const auto is_last_axis_vector = Predicate(
        [](const ov::Output<ov::Node>& out) {
            const auto& shape = out.get_shape();
            return ov::shape_size(shape) == shape.back();
        },
        "is_last_axis_vector");
    // A trailing Convert is taken in only if it is the sole consumer of the Multiply, so RMS writes the converted type
    // directly; otherwise the Multiply itself is the root and keeps its type.
    const auto converts_sole_consumer = Predicate(
        [](const ov::Output<ov::Node>& out) {
            return out.get_node()->get_input_source_output(0).get_target_inputs().size() == 1;
        },
        "converts_sole_consumer");

    auto x_m = any_input();
    auto x_convert_m = optional<v0::Convert>({x_m});
    const Attributes numpy{{"auto_broadcast", "numpy"}};
    // "N" binds the normalized dimension, so gamma can be matched against it.
    auto power_m = wrap_type<v1::Power>({x_convert_m, wrap_type<v0::Constant>(value_matches("2"))}, shape_matches("..., N"), numpy);
    auto sum_m =
        wrap_type<v1::ReduceSum>({power_m, wrap_type<v0::Constant>()}, consumers_count(1) && reduces_single_axis(-1) && fits_rms_kernel, {{"keep_dims", true}});
    auto eps_m = wrap_type<v0::Constant>(is_uniform_constant(false));
    auto add_eps_m = wrap_type<v1::Add>({sum_m, eps_m}, consumers_count(1), numpy);
    auto sqrt_m = wrap_type<v0::Sqrt>({add_eps_m}, consumers_count(1));
    auto div_m = wrap_type<v1::Divide>({x_convert_m, sqrt_m}, consumers_count(1), numpy);
    auto gamma_m = any_input(has_static_shape() && shape_matches("..., N") && is_last_axis_vector);
    auto mul_m = wrap_type<v1::Multiply>({div_m, gamma_m}, numpy);
    auto root_m = optional<v0::Convert>({mul_m}, converts_sole_consumer);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        const auto root = m.get_match_root();
        if (transformation_callback(root)) {
            return false;
        }
        if (pm.at(mul_m).get_partial_shape() != pm.at(div_m).get_partial_shape()) {
            return false;
        }

        float eps = 0.F;
        OPENVINO_ASSERT(ov::op::util::get_single_value(ov::as_type_ptr<v0::Constant>(pm.at(eps_m).get_node_shared_ptr()), eps),
                        "L2NormToRMSLastAxis: eps is expected to be a uniform Constant");
        const auto n = static_cast<double>(m.get_symbols().at("N").i());
        const auto& gamma = pm.at(gamma_m);

        // x / sqrt(sum + eps) * gamma == x / sqrt(mean + eps / n) * (gamma / sqrt(n))
        auto inv_sqrt_n = v0::Constant::create(gamma.get_element_type(), {}, {1.0 / std::sqrt(n)});
        auto scaled_gamma = ov::op::util::make_try_fold<v1::Multiply>(gamma, inv_sqrt_n);
        auto rms = std::make_shared<ov::op::internal::RMS>(pm.at(x_m), scaled_gamma, eps / static_cast<float>(n), root->get_output_element_type(0));

        rms->set_friendly_name(root->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), {scaled_gamma, rms});
        ov::replace_node(root, rms);
        return true;
    };

    auto m = std::make_shared<Matcher>(root_m, "L2NormToRMSLastAxis");
    register_matcher(m, callback);
}

L2NormToRMSChannelAxis::L2NormToRMSChannelAxis() {
    using namespace ov::pass::pattern;
    using ov::pass::operator|;

    auto x_m = any_input((rank_equals(4) || rank_equals(5)) && has_static_dim(1) && type_matches_any({ov::element::f16, ov::element::bf16, ov::element::f32}));
    auto reduce_m = wrap_type<v4::ReduceL2>({x_m, wrap_type<v0::Constant>()}, consumers_count(1) && reduces_single_axis(1), {{"keep_dims", true}});
    const Attributes numpy{{"auto_broadcast", "numpy"}};

    auto clamp_m = wrap_type<v0::Clamp>({reduce_m}, consumers_count(1) && is_eps_clamp());
    auto eps_const_m = wrap_type<v0::Constant>();
    auto max_m = wrap_type<v1::Maximum>({reduce_m, optional<v0::Convert>({eps_const_m})}, consumers_count(1) && is_negligible_eps_max(), numpy);
    auto clip_m = optional<v0::Convert>({wrap_type<v0::Constant>(is_no_op_clip())}, is_no_op_clip());
    auto min_m = optional<v1::Minimum>({max_m, clip_m}, consumers_count(1), numpy);
    auto floor_m = clamp_m | min_m;

    auto target_m = wrap_type<v0::ShapeOf, v3::ShapeOf>({x_m}) | wrap_type<v0::Constant>();
    auto bcast_m = wrap_type<v3::Broadcast>({floor_m, target_m}) | wrap_type<v1::Broadcast, v3::Broadcast>({floor_m, target_m, any_input()});
    auto norm_m = bcast_m | floor_m;

    // x / norm, or x * norm^-1 as ConvertDivide leaves it.
    auto div_m = wrap_type<v1::Divide>({x_m, norm_m}, numpy);
    auto inv_norm_m = wrap_type<v1::Power>({norm_m, optional<v0::Convert>({wrap_type<v0::Constant>(value_matches("-1"))})}, consumers_count(1), numpy);
    auto mul_inv_m = wrap_type<v1::Multiply>({x_m, inv_norm_m}, numpy);
    auto normalized_m = div_m | mul_inv_m;

    // Trailing affine chain folded into gamma: [* s] [* gamma] [+ 0], scalar or per-channel factors.
    auto scale_m = any_input();
    auto scaled_m = optional<v1::Multiply>({normalized_m, scale_m}, applies_per_channel(), numpy);
    auto gamma_m = any_input();
    auto weighted_m = optional<v1::Multiply>({scaled_m, gamma_m}, applies_per_channel(), numpy);
    auto zero_m = optional<v0::Convert>({wrap_type<v0::Constant>(is_uniform_constant(true))});
    auto root_m = optional<v1::Add>({weighted_m, zero_m}, applies_per_channel(), numpy);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        const auto root = m.get_match_root();
        if (transformation_callback(root)) {
            return false;
        }
        const auto& x = pm.at(x_m);
        if (root->get_output_partial_shape(0) != x.get_partial_shape()) {
            return false;
        }
        // Every matched node but the root must feed the chain only, or it would stay alive next to RMS. Any of the
        // tail ops can be the root, which may have any consumers, so this is checked here rather than in the pattern.
        ov::NodeVector matched;
        for (const auto& label : {reduce_m, clamp_m, max_m, min_m, bcast_m, div_m, inv_norm_m, mul_inv_m, scaled_m, weighted_m, root_m}) {
            if (pm.count(label)) {
                matched.push_back(pm.at(label).get_node_shared_ptr());
                if (matched.back() != root && matched.back()->get_output_target_inputs(0).size() != 1) {
                    return false;
                }
            }
        }
        const auto channels = static_cast<size_t>(x.get_partial_shape()[1].get_length());

        double eps = 0.0;
        if (pm.count(clamp_m)) {
            eps = ov::as_type_ptr<v0::Clamp>(pm.at(clamp_m).get_node_shared_ptr())->get_min();
        } else {
            const auto value = scalar_value(pm.at(eps_const_m));
            OPENVINO_ASSERT(value, "L2NormToRMSChannelAxis: eps is expected to be a scalar Constant");
            eps = *value;
        }

        // x / max(||x||, eps) * s * gamma == x * rsqrt(max(mean(x^2), eps^2 / C)) * (s * gamma / sqrt(C))
        //                                ~= RMS(x, s * gamma / sqrt(C), eps^2 / C)
        // gamma' is built with OV ops and folded when s and gamma are constant; it is computed in f32 and converted
        // once, so a f16 model does not round every factor separately.
        ov::NodeVector new_nodes;
        auto add_node = [&new_nodes](const std::shared_ptr<ov::Node>& node) {
            new_nodes.push_back(node);
            return node;
        };
        ov::Output<ov::Node> fused_gamma = add_node(v0::Constant::create(ov::element::f32, {}, {1.0 / std::sqrt(static_cast<double>(channels))}));
        for (const auto& label : {scale_m, gamma_m}) {
            if (!pm.count(label)) {
                continue;
            }
            ov::Output<ov::Node> factor = pm.at(label);
            if (const auto constant = ov::util::get_constant_from_source(factor)) {
                factor = constant;  // looks through decompression Converts
            }
            if (factor.get_element_type() != ov::element::f32) {
                factor = add_node(ov::op::util::make_try_fold<v0::Convert>(factor, ov::element::f32));
            }
            fused_gamma = add_node(ov::op::util::make_try_fold<v1::Multiply>(fused_gamma, factor));
        }
        const auto channels_dim = static_cast<int64_t>(channels);
        fused_gamma = add_node(ov::op::util::make_try_fold<v1::Reshape>(fused_gamma, v0::Constant::create(ov::element::i64, {1}, {-1}), false));
        if (fused_gamma.get_partial_shape() != ov::PartialShape{channels_dim}) {  // scalar s * gamma, or neither
            fused_gamma = add_node(ov::op::util::make_try_fold<v3::Broadcast>(fused_gamma, v0::Constant::create(ov::element::i64, {1}, {channels_dim})));
        }
        if (x.get_element_type() != ov::element::f32) {
            fused_gamma = add_node(ov::op::util::make_try_fold<v0::Convert>(fused_gamma, x.get_element_type()));
        }

        auto rms = std::make_shared<ov::op::internal::RMS>(x, fused_gamma, eps * eps / static_cast<double>(channels), root->get_output_element_type(0), 1);
        new_nodes.push_back(rms);

        rms->set_friendly_name(root->get_friendly_name());
        ov::copy_runtime_info(matched, new_nodes);
        ov::replace_node(root, rms);
        return true;
    };

    auto m = std::make_shared<Matcher>(root_m, "L2NormToRMSChannelAxis");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
