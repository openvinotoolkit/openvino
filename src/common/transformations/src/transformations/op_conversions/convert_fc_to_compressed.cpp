// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_fc_to_compressed.hpp"

#include <memory>
#include <tuple>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "transformations/pattern_blocks/compressed_weights_block.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;

namespace ov::pass {

std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>>
ConvertFullyConnectedToFullyConnectedCompressed::process_compressed_weights(
    const std::shared_ptr<pattern::op::CompressedWeightsBlock>& weights_block,
    const pattern::PatternValueMap& pattern_map,
    bool convert_u4zp_to_u8,
    bool has_transpose,
    bool grouped,
    bool batched_weights,
    std::vector<std::shared_ptr<ov::Node>>& result_nodes,
    bool enable_parameter_weights) {
    const size_t final_weights_rank = batched_weights ? 3 : 2;

    // Constant weights/params: fold the group dims by materializing a reshaped Constant.
    auto combine_groups_constant = [has_transpose, grouped, final_weights_rank](
                                       const std::shared_ptr<v0::Constant>& constant) -> std::shared_ptr<ov::Node> {
        const auto& current_shape = constant->get_shape();
        if (current_shape.size() <= final_weights_rank) {
            return constant;
        }

        OPENVINO_ASSERT(current_shape.size() == final_weights_rank + 1);
        ov::Shape new_shape(current_shape.begin(), current_shape.begin() + final_weights_rank);
        if (has_transpose || !grouped) {
            // [n_groups, group_size, OC] -> [IC, OC]
            const auto& n_groups = *(current_shape.rbegin() + 2);
            const auto& group_size = *(current_shape.rbegin() + 1);
            const auto& OC = *(current_shape.rbegin());
            auto& new_IC = *(new_shape.rbegin() + 1);
            auto& new_OC = *(new_shape.rbegin());
            new_IC = n_groups * group_size;
            new_OC = OC;
        } else {
            // [OC, n_groups, group_size] -> [OC, IC]
            const auto& n_groups = *(current_shape.rbegin() + 1);
            const auto& group_size = *(current_shape.rbegin());
            const auto& OC = *(current_shape.rbegin() + 2);
            auto& new_OC = *(new_shape.rbegin() + 1);
            auto& new_IC = *new_shape.rbegin();
            new_OC = OC;
            new_IC = n_groups * group_size;
        }
        auto new_constant = std::make_shared<v0::Constant>(*constant, new_shape);
        // Propagate plain "otd_bin_offset" entry (used by GPU OTD when WCA is lost)
        auto otd_it = constant->get_rt_info().find("otd_bin_offset");
        if (otd_it != constant->get_rt_info().end()) {
            new_constant->get_rt_info()["otd_bin_offset"] = otd_it->second;
        }
        return new_constant;
    };

    // Parameter weights/params: shapes are only known at runtime, so fold the group dims with a
    // Reshape instead of materializing a Constant.
    auto combine_groups_params = [has_transpose, grouped, final_weights_rank, &result_nodes](
                                     const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::Node> {
        const auto& ps = node->get_output_partial_shape(0);
        // The matcher callback already rejects dynamic weights/scale/zero-point shapes, so a
        // Parameter reaching this point must be static.
        OPENVINO_ASSERT(ps.is_static(), "Parameter input must have a static shape in combine_groups");
        if (ps.size() <= final_weights_rank) {
            // Not grouped: no group dim to fold.
            return node;
        }
        OPENVINO_ASSERT(ps.size() == final_weights_rank + 1, "Unexpected rank for grouped Parameter in combine_groups");
        ov::Shape current_shape = ps.to_shape();
        ov::Shape new_shape(current_shape.begin(), current_shape.begin() + final_weights_rank);
        if (has_transpose || !grouped) {
            new_shape[new_shape.size() - 2] =
                current_shape[current_shape.size() - 3] * current_shape[current_shape.size() - 2];
            new_shape[new_shape.size() - 1] = current_shape[current_shape.size() - 1];
        } else {
            new_shape[new_shape.size() - 2] = current_shape[current_shape.size() - 3];
            new_shape[new_shape.size() - 1] =
                current_shape[current_shape.size() - 2] * current_shape[current_shape.size() - 1];
        }
        auto shape_const = v0::Constant::create(ov::element::i64, {new_shape.size()}, new_shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(node, shape_const, false);
        result_nodes.push_back(shape_const);
        result_nodes.push_back(reshape);
        return reshape;
    };

    // Without parameter weights every input is a Constant. With parameter weights an input may be
    // a Parameter (dynamic shapes -> runtime Reshape) or still a Constant (fold at build time).
    auto combine_groups = [&combine_groups_constant, &combine_groups_params, enable_parameter_weights](
                              const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::Node> {
        auto constant = ov::as_type_ptr<v0::Constant>(node);
        if (!enable_parameter_weights || constant) {
            return combine_groups_constant(constant);
        }
        return combine_groups_params(node);
    };

    auto convert_u4const_to_u8 =
        [convert_u4zp_to_u8, enable_parameter_weights](std::shared_ptr<ov::Node> node) -> std::shared_ptr<ov::Node> {
        // Without parameter weights the zero-point must be a Constant; a non-constant here is unexpected.
        OPENVINO_ASSERT(enable_parameter_weights || ov::as_type_ptr<v0::Constant>(node),
                        "A non-constant zero-point is only expected when parameter weights are enabled");
        if (node->get_output_element_type(0) != ov::element::u4 || !convert_u4zp_to_u8)
            return node;
        return std::make_shared<v0::Convert>(node, ov::element::u8);
    };

    const auto& scale =
        combine_groups(weights_block->get_anchor("mul_const", pattern_map).value().get_node_shared_ptr());
    std::shared_ptr<ov::Node> optional_zero_point = nullptr;

    const bool with_zero_point = weights_block->get_anchor("sub_no_convert", pattern_map) ||
                                 weights_block->get_anchor("sub_with_convert", pattern_map);
    if (with_zero_point) {
        // WA: Convert ZP to u8 for OneDNN case to avoid u4 reorder
        optional_zero_point = convert_u4const_to_u8(
            combine_groups(weights_block->get_anchor("sub_const", pattern_map).value().get_node_shared_ptr()));
    }

    std::shared_ptr<ov::Node> fc_input_b =
        combine_groups(weights_block->get_anchor("weights", pattern_map).value().get_node_shared_ptr());
    std::shared_ptr<ov::Node> fc_input_scale = scale;
    std::shared_ptr<ov::Node> fc_input_zp = optional_zero_point;

    if (has_transpose) {
        const auto& transpose = weights_block->get_anchor("transpose", pattern_map).value().get_node_shared_ptr();
        std::shared_ptr<ov::Node> transpose_const =
            weights_block->get_anchor("transpose_const", pattern_map).value().get_node_shared_ptr();

        // The matched `transpose_const` was authored for the weights tensor and may have a
        // different rank than `scale` / `zero_point` (which can be per-channel rank-1
        // constants while the weights are rank-2). Align each input's rank, then build a
        // perm that matches it; rank-1 per-channel constants are unsqueezed to rank-2
        // [N, 1] so that downstream consumers (e.g. DnnlPostOpsComposer in the CPU plugin)
        // which require rank-2/3 decompression params can prepack them. Inputs with a
        // single element are left as-is.
        // All inputs reaching this lambda are Constants (optionally wrapped in a Convert
        // injected by `convert_u4const_to_u8`), so their shapes are always static.
        auto align_and_transpose = [&](const ov::Output<ov::Node>& in) -> std::shared_ptr<ov::Node> {
            const auto& in_shape = in.get_shape();
            const auto in_rank = in_shape.size();
            if (in_rank == 0 || ov::shape_size(in_shape) == 1) {
                return in.get_node_shared_ptr();
            }
            std::shared_ptr<ov::Node> node = in.get_node_shared_ptr();
            if (in_rank == 1) {
                // Promote rank-1 per-channel constant to rank-2 [N, 1] via Unsqueeze.
                // Peel a wrapping Convert (injected by convert_u4const_to_u8 for u4 ZP)
                // so make_try_fold can collapse Unsqueeze on the underlying Constant,
                // then re-apply the Convert on the folded result.
                auto wrapping_convert = ov::as_type_ptr<v0::Convert>(node);
                auto inner = wrapping_convert ? wrapping_convert->get_input_node_shared_ptr(0) : node;
                auto axis = v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
                auto unsqueezed = ov::op::util::make_try_fold<v0::Unsqueeze>(inner, axis);
                result_nodes.push_back(unsqueezed);
                if (wrapping_convert) {
                    auto rewrapped =
                        std::make_shared<v0::Convert>(unsqueezed, wrapping_convert->get_destination_type());
                    result_nodes.push_back(rewrapped);
                    return rewrapped;
                }
                return unsqueezed;
            }
            std::shared_ptr<ov::Node> perm = transpose_const;
            if (ov::shape_size(perm->get_shape()) != in_rank) {
                std::vector<int32_t> new_order(in_rank);
                std::iota(new_order.begin(), new_order.end(), 0);
                std::swap(new_order[in_rank - 1], new_order[in_rank - 2]);
                perm = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{in_rank}, new_order);
            }
            auto transposed = transpose->clone_with_new_inputs({node->output(0), perm});
            ov::disable_constant_folding(transposed);
            result_nodes.push_back(transposed);
            return transposed;
        };

        fc_input_b = align_and_transpose(fc_input_b->output(0));
        fc_input_scale = align_and_transpose(scale->output(0));
        if (with_zero_point && ov::shape_size(optional_zero_point->output(0).get_shape()) > 1) {
            fc_input_zp = align_and_transpose(optional_zero_point->output(0));
        }
    }

    if (!with_zero_point) {
        // No zero-point: emit an empty placeholder Constant. Downstream ops detect "absent ZP"
        // via element count() == 0, not the element type. With enable_parameter_weights use the
        // weight element type instead of element::dynamic, which VCL (NPU) cannot handle.
        const auto zp_et = enable_parameter_weights
                               ? weights_block->get_anchor("weights", pattern_map).value().get_element_type()
                               : ov::element::dynamic;
        fc_input_zp = std::make_shared<v0::Constant>(zp_et, ov::Shape{0});
    }
    ov::disable_constant_folding(fc_input_zp);
    result_nodes.push_back(fc_input_zp);

    return std::make_tuple(fc_input_b, fc_input_scale, fc_input_zp);
}

ConvertFullyConnectedToFullyConnectedCompressed::ConvertFullyConnectedToFullyConnectedCompressed(
    const std::vector<ov::element::Type>& supported_activation_types,
    const std::vector<ov::element::Type>& supported_weights_types,
    SupportsPredicate supports_config,
    bool convert_u4zp_to_u8,
    bool enable_parameter_weights) {
    auto weights_block = std::make_shared<pattern::op::CompressedWeightsBlock>(supported_weights_types,
                                                                               std::set<size_t>{2},
                                                                               enable_parameter_weights);
    auto activation = pattern::any_input(pattern::type_matches_any(supported_activation_types));
    auto bias = pattern::any_input();
    auto fully_connected = pattern::wrap_type<ov::op::internal::FullyConnected>({activation, weights_block, bias});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto fc =
            ov::as_type_ptr<ov::op::internal::FullyConnected>(pattern_map.at(fully_connected).get_node_shared_ptr());
        if (!fc || transformation_callback(fc)) {
            return false;
        }

        bool has_transpose = weights_block->get_anchor("transpose", pattern_map).has_value();
        // Weights/scale are static here: constants always are, and CompressedWeightsBlock requires
        // static shapes for the parameter-weights case (has_static_shape predicate).
        const auto& weights_pshape = fc->get_input_partial_shape(1);
        const auto& scale_pshape = weights_block->get_anchor("mul_const", pattern_map).value().get_partial_shape();
        const auto weights_shape = weights_pshape.to_shape();
        bool batched_weights = weights_shape.size() == 3 && weights_shape[0] > 1;
        const auto scale_shape = scale_pshape.to_shape();
        bool grouped = scale_shape.size() == weights_shape.size() + 1;
        ov::NodeVector result_nodes;
        const auto [fc_input_b, fc_input_scale, fc_input_zp] = process_compressed_weights(weights_block,
                                                                                          pattern_map,
                                                                                          convert_u4zp_to_u8,
                                                                                          has_transpose,
                                                                                          grouped,
                                                                                          batched_weights,
                                                                                          result_nodes,
                                                                                          enable_parameter_weights);

        auto new_fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(pattern_map.at(activation),
                                                                                   fc_input_b,
                                                                                   pattern_map.at(bias),
                                                                                   fc_input_scale,
                                                                                   fc_input_zp,
                                                                                   fc->get_output_type());

        const size_t IC = *(weights_shape.rbegin());
        const size_t OC = *(weights_shape.rbegin() + 1);
        const size_t G = grouped ? (has_transpose ? *(scale_shape.rbegin() + 2) : *(scale_shape.rbegin() + 1)) : 1;
        if (supports_config && !supports_config(new_fc, IC, OC, G))
            return false;

        result_nodes.push_back(new_fc);
        new_fc->set_friendly_name(fc->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(fc, new_fc);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(fully_connected, "ConvertFullyConnectedToFullyConnectedCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
