// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "l2_norm_to_rms.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_set>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {
// The ops that PropagateUpMarkToKeepInMixedPrecision passes an FP32 mark up through (propagate_through_ops in
// mark_subgraphs_to_keep_in_mixed_precision.cpp), without Convert and Constant: the mark goes up through a Convert
// only to a Constant.
bool propagates_fp32_mark_up(const ov::Node* node) {
    return ov::is_type_any_of<ov::op::v0::Squeeze,
                              ov::op::v0::Unsqueeze,
                              ov::op::v1::Reshape,
                              ov::op::util::BroadcastBase,
                              ov::op::util::BinaryElementwiseArithmetic,
                              ov::op::util::UnaryElementwiseArithmetic,
                              ov::op::v6::MVN,
                              ov::op::v0::MVN,
                              ov::op::v0::NormalizeL2,
                              ov::op::v0::Sqrt,
                              ov::op::v1::StridedSlice,
                              ov::op::v1::ReduceSum,
                              ov::op::v1::ReduceMean,
                              ov::op::v8::Slice,
                              ov::op::v1::VariadicSplit,
                              ov::op::v1::Split,
                              ov::op::v0::Concat,
                              ov::op::v0::Tile>(node);
}

// A Constant, possibly behind Converts (a compressed constant). Anything else counts as non-constant, e.g. ReadValue
// (its state changes at run time, even with a Constant initializer).
bool is_constant(const ov::Output<ov::Node>& output) {
    const ov::Node* node = output.get_node();
    while (ov::is_type<ov::op::v0::Convert>(node)) {
        node = node->get_input_node_ptr(0);
    }
    return ov::is_type<ov::op::v0::Constant>(node);
}

// Whether an FP32 mark on the norm would reach a residual connection: walks up from x through the ops that the mark
// propagates up through (see propagates_fp32_mark_up) and looks for elementwise arithmetic of two non-constant inputs,
// e.g. the residual Add. A constant operand (a bias or a scale) is not a residual connection. A real-typed model input
// counts as one: it may carry a residual stream (e.g. a model split per layer), and the decomposed norm kept it in
// FP32 as well.
bool reaches_residual_connection(const ov::Output<ov::Node>& x) {
    std::vector<const ov::Node*> to_visit{x.get_node()};
    std::unordered_set<const ov::Node*> visited;
    while (!to_visit.empty()) {
        const ov::Node* node = to_visit.back();
        to_visit.pop_back();
        if (!visited.insert(node).second) {
            continue;
        }
        if (ov::is_type<ov::op::v0::Parameter>(node)) {
            if (node->get_output_element_type(0).is_real()) {
                return true;
            }
            continue;
        }
        if (!propagates_fp32_mark_up(node)) {
            continue;
        }
        size_t non_constant_inputs = 0;
        for (const auto& input : node->input_values()) {
            // The mark propagates up through real-typed inputs only.
            if (!input.get_element_type().is_real() || is_constant(input)) {
                continue;
            }
            ++non_constant_inputs;
            to_visit.push_back(input.get_node());
        }
        if (non_constant_inputs > 1 && ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(node)) {
            return true;
        }
    }
    return false;
}

// Keeps in FP32 the Converts that dequantize a non-constant gamma, e.g. Convert(u16 model input -> f32): the FP32
// mark on the scaled gamma propagates up to them but not through them (PropagateUpMarkToKeepInMixedPrecision goes
// through a Convert only to a Constant), so ConvertPrecision would convert them to f16, and u16 65535 is inf in f16.
void keep_gamma_dequantization_in_fp32(const ov::Output<ov::Node>& gamma) {
    std::vector<ov::Node*> to_visit{gamma.get_node()};
    std::unordered_set<ov::Node*> visited;
    while (!to_visit.empty()) {
        ov::Node* node = to_visit.back();
        to_visit.pop_back();
        if (!visited.insert(node).second) {
            continue;
        }
        if (ov::is_type<ov::op::v0::Convert>(node)) {
            // Same rule as PropagateUpMarkToKeepInMixedPrecision: it marks a Convert only if its input is a Constant.
            if (!ov::is_type<ov::op::v0::Constant>(node->get_input_node_ptr(0))) {
                ov::disable_conversion(node->shared_from_this(), ov::element::f16);
            }
            continue;
        }
        if (!propagates_fp32_mark_up(node)) {
            continue;
        }
        for (const auto& input : node->input_values()) {
            if (input.get_element_type().is_real()) {
                to_visit.push_back(input.get_node());
            }
        }
    }
}
}  // namespace

L2NormToRMS::L2NormToRMS(size_t max_work_group_size) {
    using namespace ov::pass::pattern;

    auto x = any_input();
    auto x_convert = optional<ov::op::v0::Convert>({x});
    auto power = wrap_type<ov::op::v1::Power>({x_convert, wrap_type<ov::op::v0::Constant>(value_matches("2"))});
    auto axes = wrap_type<ov::op::v0::Constant>();
    auto sum = wrap_type<ov::op::v1::ReduceSum>({power, axes}, consumers_count(1));
    auto eps = wrap_type<ov::op::v0::Constant>();
    auto add_eps = wrap_type<ov::op::v1::Add>({sum, eps}, consumers_count(1));
    auto sqrt = wrap_type<ov::op::v0::Sqrt>({add_eps}, consumers_count(1));
    auto div = wrap_type<ov::op::v1::Divide>({x_convert, sqrt}, consumers_count(1));
    auto gamma = any_input();
    auto mul = wrap_type<ov::op::v1::Multiply>({div, gamma});

    ov::matcher_pass_callback callback = [this, x_convert, power, sum, axes, add_eps, sqrt, div, gamma, eps, x, mul, max_work_group_size](Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto sum_node = ov::as_type_ptr<ov::op::v1::ReduceSum>(pm.at(sum).get_node_shared_ptr());
        if (!sum_node || !sum_node->get_keep_dims() || transformation_callback(sum_node)) {
            return false;
        }

        // Only the last axis, and only a static one: N goes into the rewritten gamma and eps.
        const auto& in_shape = sum_node->get_input_partial_shape(0);
        if (in_shape.rank().is_dynamic() || in_shape[in_shape.size() - 1].is_dynamic()) {
            return false;
        }
        const auto rank = static_cast<int64_t>(in_shape.size());
        // The GPU RMS kernel has bfyx and bfzyx implementations only.
        if (rank > 5) {
            return false;
        }
        auto axes_values = ov::as_type_ptr<ov::op::v0::Constant>(pm.at(axes).get_node_shared_ptr())->cast_vector<int64_t>();
        if (axes_values.size() != 1 || (axes_values[0] != -1 && axes_values[0] != rank - 1)) {
            return false;
        }
        const auto n = static_cast<size_t>(in_shape[rank - 1].get_length());

        // Same limit RMSFusion applies on GPU: the kernel handles a row with one work group.
        const size_t vec_size = 8;
        if (n / vec_size > max_work_group_size) {
            return false;
        }

        // RMS scales along the normalized axis, so gamma must be {..., 1, n} with no more dims than x. A gamma
        // with n elements along another axis, e.g. {n, 1}, broadcasts per row instead.
        const auto& gamma_out = pm.at(gamma);
        const auto& gamma_shape = gamma_out.get_partial_shape();
        if (gamma_shape.is_dynamic()) {
            return false;
        }
        const auto gamma_dims = gamma_shape.to_shape();
        if (gamma_dims.empty() || gamma_dims.size() > static_cast<size_t>(rank) || gamma_dims.back() != n || ov::shape_size(gamma_dims) != n) {
            return false;
        }

        float eps_value = 0.F;
        if (!ov::op::util::get_single_value(ov::as_type_ptr<ov::op::v0::Constant>(pm.at(eps).get_node_shared_ptr()), eps_value)) {
            return false;
        }

        const auto x_out = pm.at(x);
        // Nodes are visited in topological order, so the match lands on the Multiply before its Convert is
        // seen; take a lone trailing Convert in as well so RMS writes the converted type directly.
        std::shared_ptr<ov::Node> root = pm.at(mul).get_node_shared_ptr();
        const auto& mul_users = root->get_output_target_inputs(0);
        if (mul_users.size() == 1) {
            if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(mul_users.begin()->get_node()->shared_from_this())) {
                root = convert;
            }
        }
        const auto output_type = root->get_output_element_type(0);

        // The nodes that the fusion removes: the matched norm and the absorbed trailing Convert, if any.
        std::vector<std::shared_ptr<ov::Node>> removed_nodes{root};
        for (const auto& pattern_node : {x_convert, power, sum, add_eps, sqrt, div, mul}) {
            const auto it = pm.find(pattern_node);
            if (it != pm.end() && it->second.get_node_shared_ptr() != root) {
                removed_nodes.push_back(it->second.get_node_shared_ptr());
            }
        }
        // A removed node may already be marked to stay in FP32, e.g. by the model's rt_info, also only for its own
        // source type (disable_conversion(node, f32, f16)), which ConvertPrecision honors. copy_runtime_info does not
        // copy this mark, so the RMS gets it below.
        const bool removed_node_marked = std::any_of(removed_nodes.begin(), removed_nodes.end(), [](const std::shared_ptr<ov::Node>& node) {
            return ov::is_conversion_disabled(node, node->get_output_element_type(0), ov::element::f16);
        });

        // x / sqrt(sum + eps) * gamma == x / sqrt(mean + eps / n) * (gamma / sqrt(n))
        auto inv_sqrt_n = ov::op::v0::Constant::create(gamma_out.get_element_type(), {}, {1.0 / std::sqrt(static_cast<double>(n))});
        auto scaled_gamma = std::make_shared<ov::op::v1::Multiply>(gamma_out, inv_sqrt_n);
        auto rms = std::make_shared<ov::op::internal::RMS>(x_out, scaled_gamma, eps_value / static_cast<float>(n), output_type);

        rms->set_friendly_name(root->get_friendly_name());
        auto rt_info_sources = m.get_matched_nodes();
        if (root != pm.at(mul).get_node_shared_ptr()) {
            rt_info_sources.push_back(root);  // the absorbed trailing Convert is not part of the match
        }
        ov::copy_runtime_info(rt_info_sources, {scaled_gamma, rms});
        ov::replace_node(root, rms);

        // The gamma may be dequantized from u16 values above the f16 range (e.g. 65535): keep it in FP32.
        ov::disable_conversion(scaled_gamma, ov::element::f16);
        keep_gamma_dequantization_in_fp32(gamma_out);
        // Keeps the residual stream in FP32, as MarkDivWithEps did for the decomposed norm: ConvertPrecision
        // propagates this mark up to the residual Adds. Other norms may run in f16, unless a removed node was marked.
        if (removed_node_marked || reaches_residual_connection(x_out)) {
            ov::disable_conversion(rms, ov::element::f16);
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(mul, "L2NormToRMS");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
