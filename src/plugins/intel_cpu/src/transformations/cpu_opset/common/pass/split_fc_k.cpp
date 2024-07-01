// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/constant_folding.hpp"
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <transformations/utils/utils.hpp>
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"

#include "split_fc_k.hpp"

#include "itt.hpp"

static int weightsThreshold() {
    static int result = std::getenv("SPLIT_THRESHOLD") ? std::stoi(std::getenv("SPLIT_THRESHOLD")) : 6600000;
    return result;
}

ov::intel_cpu::SplitFCbyK::SplitFCbyK(int sub_stream_num) {
    MATCHER_SCOPE(SplitFCbyK);
    auto fc_m = ov::pass::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& fc_node = pattern_map.at(fc_m).get_node_shared_ptr();
        auto& rt_info = fc_node->get_rt_info();
        if (rt_info.count("split")) {
            return false;
        }

        const auto src_item = fc_node->get_input_node_shared_ptr(0);
        const auto fc_weight_node = fc_node->get_input_node_shared_ptr(1);

        // TODO: support transpose
        if (ov::is_type<ov::op::v1::Transpose>(fc_weight_node)) {
            std::cout << "SplitFCbyK: Transpose on weights is not supported" << "\n";
            return false;
        }

        // needn't to split fc when the dim is 0.
        const auto& wgt_shape = fc_weight_node->get_shape();
        const size_t split_dim = wgt_shape.size() - 1;
        // split happens on the second dimension.
        auto split_dim_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, split_dim);

        // std::cout << "SplitFCbyK transformation check" << "\n";
        // weight shape size 660000 is a trade-off value, which is summarized and verified by LLMs.
        if (wgt_shape[split_dim] <= 1 || ov::shape_size(wgt_shape) < weightsThreshold()) {
            std::cout << "Heuristic not met. split dim: " << wgt_shape[split_dim] <<
                " shape_size: " << ov::shape_size(wgt_shape) << "\n";
            return false;
        }

        // parts will be splited according the sub stream num.
        int split_num = sub_stream_num + 1;
        std::cout << "SplitFCbyK transformation. Heuristic met:" << ov::shape_size(wgt_shape) << "! " <<  fc_node
                  << " split into parts = " << split_num << "\n";

        auto split_on_parts = [](int len, int n) {
            int average = len / n;
            std::vector<int> parts(n, average);
            parts.back() = len - average * (n - 1);
            return parts;
        };

        // 1. If the model is INT4 format, split the INT4 pattern for the FuseFCAndWeightsDecompression.
        // 2. If the model is NOT INT4 format, split the weight.
        std::vector<ov::Output<ov::Node>> wgt_node_vec(split_num);
        if (ov::is_type<ov::op::v1::Multiply>(fc_weight_node) || ov::is_type<ov::op::v1::Reshape>(fc_weight_node)) {
            // INT4 model should consider two patterns, including with Reshape Node and without Reshape Node.
            const auto reshape_node = ov::as_type_ptr<ov::op::v1::Reshape>(fc_weight_node);
            const auto multiply_node = reshape_node ? reshape_node->get_input_node_shared_ptr(0) : fc_weight_node;
            if (!ov::is_type<ov::op::v1::Multiply>(multiply_node)) {
                return false;
            }
            auto multiply_pattern = multiply_node->get_input_node_shared_ptr(1);
            if (!ov::is_type<ov::op::v0::Constant>(multiply_pattern)) {
                return false;
            }

            auto subtract_node = multiply_node->get_input_node_shared_ptr(0);
            if (!ov::is_type<ov::op::v1::Subtract>(subtract_node)) {
                return false;
            }
            auto convert_node1 = subtract_node->get_input_node_shared_ptr(1);
            if (!ov::is_type<ov::op::v0::Convert>(convert_node1)) {
                return false;
            }
            auto convert_node1_const = ov::as_type_ptr<ov::op::v0::Constant>(convert_node1->get_input_node_shared_ptr(0));
            if (!convert_node1_const) {
                return false;
            }
            auto convert_node0 = subtract_node->get_input_node_shared_ptr(0);
            if (!ov::is_type<ov::op::v0::Convert>(convert_node0)) {
                return false;
            }
            auto wgt_item = convert_node0->get_input_node_shared_ptr(0);
            auto cvt_prec = convert_node0->get_element_type();

            auto split_dim_range = wgt_item->get_shape()[split_dim];

            const auto& multiply_shape = multiply_pattern->get_shape();
            bool need_to_split_multiply = ov::shape_size(multiply_shape) > 1 &&
                                         split_dim < multiply_shape.size() &&
                                         multiply_shape[split_dim] == split_dim_range;

            const auto& convert_node1_shape = convert_node1->get_shape();
            bool need_to_split_convert = ov::shape_size(convert_node1_shape) > 1 &&
                                         split_dim < convert_node1_shape.size() &&
                                         convert_node1_shape[split_dim] == split_dim_range;

            // We should use VariadicSplit to split the input for FC.
            std::vector<std::vector<int32_t>> split_reshape_pattern_vec(split_num);
            auto fc_dim_vec = split_on_parts(split_dim_range, split_num);
            auto split_length = ov::op::v0::Constant::create<int32_t>(ov::element::i32, ov::Shape{static_cast<size_t>(split_num)}, fc_dim_vec);

            auto split_constants = [&](const std::shared_ptr<ov::Node>& constant) {
                static const std::set<ov::element::Type> unsupported_by_split_element_types{ov::element::u4, ov::element::i4, ov::element::nf4};
                const auto& constant_precision = constant->get_output_element_type(0);
                if (!unsupported_by_split_element_types.count(constant_precision)) {
                    auto split = std::make_shared<ov::op::v1::VariadicSplit>(constant, split_dim_node, split_length);
                    return split->outputs();
                }

                auto convert = std::make_shared<ov::op::v0::Convert>(constant, ov::element::i8);
                auto split = std::make_shared<ov::op::v1::VariadicSplit>(convert, split_dim_node, split_length);
                ov::OutputVector res(split->get_output_size());
                for (size_t i = 0; i < split->get_output_size(); ++i) {
                    res[i] = std::make_shared<ov::op::v0::Convert>(split->output(i), constant_precision);
                }
                return res;
            };

            auto split_wgts = split_constants(wgt_item);
            ov::OutputVector split_muls;
            if (need_to_split_multiply) {
                split_muls = split_constants(multiply_pattern);
            }
            ov::OutputVector split_cvts;
            if (need_to_split_convert) {
                split_cvts = split_constants(convert_node1_const);
            }

            if (reshape_node) {
                auto reshape_pattern = reshape_node->get_input_node_shared_ptr(1);
                auto reshape_const = ov::as_type_ptr<ov::op::v0::Constant>(reshape_pattern);
                if (!reshape_const) {
                    return false;
                }
                const auto reshape_vec = reshape_const->cast_vector<int32_t>();
                for (int i = 0; i < split_num; ++i) {
                    split_reshape_pattern_vec[i] = {fc_dim_vec[i], reshape_vec[1]};
                }
            }

            std::vector<ov::Output<ov::Node>> zp_const_vec(split_num);
            for (int i = 0; i < split_num; ++i) {
                zp_const_vec[i] = need_to_split_convert ? split_cvts[i] : convert_node1_const->clone_with_new_inputs({});
            }

            std::vector<ov::Output<ov::Node>> mul_const_vec(split_num);
            for (int i = 0; i < split_num; ++i) {
                mul_const_vec[i] = need_to_split_multiply ? split_muls[i] : multiply_pattern->clone_with_new_inputs({});
            }

            for (int i = 0; i < split_num; ++i) {
                auto sub_parent0 = std::make_shared<ov::op::v0::Convert>(split_wgts[i], cvt_prec);
                auto sub_parent1 = std::make_shared<ov::op::v0::Convert>(zp_const_vec[i], cvt_prec);
                ov::pass::disable_constant_folding(sub_parent0);
                ov::pass::disable_constant_folding(sub_parent1);
                auto sub_node = std::make_shared<ov::op::v1::Subtract>(sub_parent0, sub_parent1);

                auto mul_node = std::make_shared<ov::op::v1::Multiply>(sub_node, mul_const_vec[i]);
                if (reshape_node) {
                    auto reshape_pattern = ov::op::v0::Constant::create<int32_t>(ov::element::i32, ov::Shape{2}, split_reshape_pattern_vec[i]);
                    wgt_node_vec[i] = std::make_shared<ov::op::v1::Reshape>(mul_node, reshape_pattern, reshape_node->get_special_zero());
                } else {
                    wgt_node_vec[i] = mul_node;
                }
            }
        } else {
            // get input
            auto wgt_item = fc_node->get_input_node_shared_ptr(1);

            // split weight
            auto split_dim_range = wgt_item->get_shape()[split_dim];

            // We should use VariadicSplit to split input for FC.
            auto fc_dim_vec = split_on_parts(split_dim_range, split_num);
            auto split_length = ov::op::v0::Constant::create<int32_t>(ov::element::i32, ov::Shape{static_cast<size_t>(split_num)}, fc_dim_vec);
            auto split_wgts = std::make_shared<ov::op::v1::VariadicSplit>(wgt_item,
                                                                          split_dim_node,
                                                                          split_length);

            wgt_node_vec = split_wgts->outputs();
        }

        const PartialShape shape = fc_node->input(0).get_partial_shape();
        auto rank = shape.get_max_shape().size();
        // @todo check if static
        const Dimension K = shape[rank - 1];
        Dimension K_split_dim = K / split_num;

        std::vector<Dimension> split_fc_dims;
        for (int i = 0; i < rank; i++) {
            split_fc_dims.push_back(shape[i]);
        }
        split_fc_dims.push_back(K_split_dim);
        const PartialShape split_fc_shape(split_fc_dims);

        const auto K_split = K_split_dim.get_length();
        std::vector<std::shared_ptr<Node>> fc_node_vec(split_num);
        std::vector<std::shared_ptr<Node>> ss_node_vec(split_num);
        for (int i = 0; i < split_num; i++) {
            int64_t offset = K_split * (i + 1);
            std::vector<int64_t> begin(rank, K_split * i); // Start from {0, 0}
            std::vector<int64_t> end(rank, -1);
            end.back() = offset;
            std::vector<int64_t> strides(rank, 1); // Stride of 1 for both dimensions

            // Optionally, you can define masks if necessary
            std::vector<int64_t> begin_mask(rank, 1); // Include all elements starting from 'begin'
            begin_mask.back() = 0;
            std::vector<int64_t> end_mask(rank, 1);   // Include all elements up to 'end'
            end_mask.back() = 0;

            // Create the StridedSlice node
            ss_node_vec[i] = std::make_shared<ov::op::v1::StridedSlice>(
                src_item,
                ov::op::v0::Constant::create(ov::element::i64, {begin.size()}, begin),
                ov::op::v0::Constant::create(ov::element::i64, {end.size()}, end),
                ov::op::v0::Constant::create(ov::element::i64, {strides.size()}, strides),
                begin_mask,
                end_mask);
            ss_node_vec[i]->get_rt_info()["split"] = true;
            if (i > 0)
                ss_node_vec[i]->get_rt_info()["other_split"] = true;

            // int64_t K_offset = K_split * i;
            // int64_t activation_stride = K_split;
            // int64_t activation_offset = K_offset;
            // std::cout << "SplitFCbyK: offset:" << K_offset << "\n";

            fc_node_vec[i] = std::make_shared<ov::intel_cpu::FullyConnectedNode>(
                // src_item,
                ss_node_vec[i],
                wgt_node_vec[i],
                fc_node->get_output_partial_shape(0).rank(),
                ov::element::undefined);
                // activation_stride,
                // activation_offset);

            fc_node_vec[i]->set_friendly_name(fc_node->get_friendly_name() + "_split_" + std::to_string(i));
            fc_node_vec[i]->get_rt_info()["split_part"] = true;
            if (i > 0)
                fc_node_vec[i]->get_rt_info()["other_split"] = true;
        }
        fc_node_vec[0]->get_rt_info()["main_split_root"] = true;
        fc_node_vec[0]->get_rt_info()["split_parts"] = fc_node_vec;
        ss_node_vec[0]->get_rt_info()["main_split"] = true;
        ss_node_vec[0]->get_rt_info()["split_parts"] = ss_node_vec;

        // @todo currently works only for split = 2.
        // split > 2 requires a chain of Add nodes
        auto add_node = std::make_shared<ov::op::v1::Add>(fc_node_vec[0], fc_node_vec[1]);
        add_node->get_rt_info()["sync_point"] = true;

        ov::replace_node_update_name(fc_node, add_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_m, matcher_name);
    this->register_matcher(m, callback);
}
