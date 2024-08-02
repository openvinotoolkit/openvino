// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/constant_folding.hpp"
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

#include "split_fc.hpp"

#include "itt.hpp"

ov::intel_cpu::SplitFC::SplitFC(int sub_stream_num) {
    MATCHER_SCOPE(SplitFC);
    auto fc_m = ov::pass::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& fc_node = pattern_map.at(fc_m).get_node_shared_ptr();
        auto& rt_info = fc_node->get_rt_info();
        if (rt_info.count("parallelDomain")) {
            return false;
        }

        const auto src_item = fc_node->get_input_node_shared_ptr(0);
        const auto fc_weight_node = fc_node->get_input_node_shared_ptr(1);

        // split happens on the first dimension.
        constexpr size_t split_dim = 0;
        auto split_dim_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, split_dim);

        // needn't to split fc when the dim is 0.
        const auto& wgt_shape = fc_weight_node->get_shape();
        // weight shape size 660000 is a trade-off value, which is summarized and verified by LLMs.
        if (wgt_shape[split_dim] <= 1 || ov::shape_size(wgt_shape) < 6600000) {
            return false;
        }

        // parts will be splited according the sub stream num.
        int split_num = sub_stream_num + 1;

        auto split_parts = [](int len, int n) {
            int average = len / n;
            std::vector<int> parts(n, average);
            parts.back() = len - average * (n - 1);
            return parts;
        };

        // TODO: support transpose
        if (ov::is_type<ov::op::v1::Transpose>(fc_weight_node)) {
            return false;
        }

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
            const auto& convert_node1_shape = convert_node1->get_shape();
            bool need_to_split_convert = ov::shape_size(convert_node1_shape) > 1 &&
                                         split_dim < convert_node1_shape.size() &&
                                         convert_node1_shape[split_dim] == split_dim_range;

            // We should use VariadicSplit to split the input for FC.
            std::vector<std::vector<int32_t>> split_reshape_pattern_vec(split_num);
            auto fc_dim_vec = split_parts(split_dim_range, split_num);
            auto split_length = ov::op::v0::Constant::create<int32_t>(ov::element::i32, ov::Shape{static_cast<size_t>(split_num)}, fc_dim_vec);

            auto split_constants = [&](const std::shared_ptr<ov::Node>& constant) {
                static const std::set<ov::element::Type> unsupported_by_split_element_types{ov::element::u4, ov::element::i4, ov::element::nf4};
                const auto& constant_precision = constant->get_output_element_type(0);
                if (unsupported_by_split_element_types.count(constant_precision) == 0) {
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
            auto split_muls = split_constants(multiply_pattern);
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

            for (int i = 0; i < split_num; ++i) {
                auto sub_parent0 = std::make_shared<ov::op::v0::Convert>(split_wgts[i], cvt_prec);
                auto sub_parent1 = std::make_shared<ov::op::v0::Convert>(zp_const_vec[i], cvt_prec);
                ov::pass::disable_constant_folding(sub_parent0);
                ov::pass::disable_constant_folding(sub_parent1);
                auto sub_node = std::make_shared<ov::op::v1::Subtract>(sub_parent0, sub_parent1);

                auto mul_node = std::make_shared<ov::op::v1::Multiply>(sub_node, split_muls[i]);
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
            auto fc_dim_vec = split_parts(split_dim_range, split_num);
            auto split_length = ov::op::v0::Constant::create<int32_t>(ov::element::i32, ov::Shape{static_cast<size_t>(split_num)}, fc_dim_vec);
            auto split_wgts = std::make_shared<ov::op::v1::VariadicSplit>(wgt_item,
                                                                          split_dim_node,
                                                                          split_length);

            wgt_node_vec = split_wgts->outputs();
        }

        // create fc Nodes according to the splited weight or splited pattern.
        std::vector<std::shared_ptr<Node>> fc_node_vec(split_num);
        for (int i = 0; i < split_num; ++i) {
            fc_node_vec[i] = fc_node->clone_with_new_inputs(ov::OutputVector{src_item, wgt_node_vec[i]});
            fc_node_vec[i]->get_rt_info()["parallelDomain"] = fc_node->get_name();
        }

        // concat all small fc for result.
        ov::NodeVector concat_args(std::move(fc_node_vec));
        // concat happens on the latest dimension.
        constexpr size_t concat_dim = -1;
        auto concat_node = std::make_shared<ov::op::v0::Concat>(concat_args, concat_dim);

        // check the shape after transformation.
        const auto& out_shape = fc_node->get_output_partial_shape(0);
        const auto& concat_shape = concat_node->get_output_partial_shape(0);
        if (concat_shape != out_shape) {
            return false;
        }
        ov::replace_node_update_name(fc_node, concat_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_m, matcher_name);
    this->register_matcher(m, callback);
}
