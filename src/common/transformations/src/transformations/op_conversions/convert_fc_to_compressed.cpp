// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_fc_to_compressed.hpp"

#include <algorithm>
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
    std::vector<std::shared_ptr<ov::Node>>& result_nodes) {
    const size_t final_weights_rank = batched_weights ? 3 : 2;
    auto combine_groups = [has_transpose, grouped, final_weights_rank](std::shared_ptr<ov::Node> node) {
        auto constant = ov::as_type_ptr<v0::Constant>(node);
        OPENVINO_ASSERT(constant != nullptr);
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
        return std::make_shared<v0::Constant>(*constant, new_shape);
    };

    auto convert_u4const_to_u8 = [convert_u4zp_to_u8](std::shared_ptr<ov::Node> node) -> std::shared_ptr<ov::Node> {
        auto constant = ov::as_type_ptr<v0::Constant>(node);
        if (constant->get_element_type() != ov::element::u4 || !convert_u4zp_to_u8)
            return std::dynamic_pointer_cast<ov::Node>(constant);
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
        if (ov::shape_size(transpose_const->get_shape()) != fc_input_b->get_output_partial_shape(0).size()) {
            std::vector<int32_t> new_order(fc_input_b->get_output_partial_shape(0).size());
            std::iota(new_order.begin(), new_order.end(), 0);
            std::swap(new_order[new_order.size() - 1], new_order[new_order.size() - 2]);
            transpose_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{new_order.size()}, new_order);
        }

        fc_input_b = transpose->clone_with_new_inputs({fc_input_b->output(0), transpose_const});
        ov::disable_constant_folding(fc_input_b);
        result_nodes.push_back(fc_input_b);
        fc_input_scale = transpose->clone_with_new_inputs({scale->output(0), transpose_const});
        ov::disable_constant_folding(fc_input_scale);
        result_nodes.push_back(fc_input_scale);
        if (with_zero_point && ov::shape_size(optional_zero_point->output(0).get_shape()) > 1) {
            fc_input_zp = transpose->clone_with_new_inputs({optional_zero_point->output(0), transpose_const});
            ov::disable_constant_folding(fc_input_zp);
            result_nodes.push_back(fc_input_zp);
        }
    }

    fc_input_zp = with_zero_point ? fc_input_zp : std::make_shared<v0::Constant>(ov::element::dynamic, ov::Shape{0});
    ov::disable_constant_folding(fc_input_zp);
    result_nodes.push_back(fc_input_zp);

    return std::make_tuple(fc_input_b, fc_input_scale, fc_input_zp);
}

ConvertFullyConnectedToFullyConnectedCompressed::ConvertFullyConnectedToFullyConnectedCompressed(
    const std::vector<ov::element::Type>& supported_activation_types,
    const std::vector<ov::element::Type>& supported_weights_types,
    SupportsPredicate supports_config,
    bool convert_u4zp_to_u8) {
    auto weights_block =
        std::make_shared<pattern::op::CompressedWeightsBlock>(supported_weights_types, std::set<size_t>{2});
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
        const auto& weights_shape = fc->get_input_shape(1);
        bool batched_weights = weights_shape.size() == 3 && weights_shape[0] > 1;
        auto scale_shape = weights_block->get_anchor("mul_const", pattern_map).value().get_shape();
        bool grouped = std::count_if(scale_shape.begin(), scale_shape.end(), [](size_t d) {
                           return d > 1;
                       }) > (batched_weights ? 2 : 1);
        ov::NodeVector result_nodes;
        const auto [fc_input_b, fc_input_scale, fc_input_zp] = process_compressed_weights(weights_block,
                                                                                          pattern_map,
                                                                                          convert_u4zp_to_u8,
                                                                                          has_transpose,
                                                                                          grouped,
                                                                                          batched_weights,
                                                                                          result_nodes);

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
