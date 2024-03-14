// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "split_fc.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/constant_folding.hpp"
#include <transformations/utils/utils.hpp>
#include "openvino/opsets/opset12.hpp"

#include "itt.hpp"

/*
    -----       -----------       -----------
    | A |   x   |    B    |   =   |    C    |
    -----       -----------       -----------
                    |   |   |
                    V   V   V
    -----       -----------       -----------
    | A |   x   | B0 | B1 |   =   | C0 | C1 |
    -----       -----------       -----------
*/
#define CHECK_NODE_VALID(type_ptr) \
    if (!type_ptr) { return false; }

ov::intel_cpu::SplitFC::SplitFC() {
    MATCHER_SCOPE(SplitFC);
    auto fc_m = ov::pass::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& fc_node = pattern_map.at(fc_m).get_node_shared_ptr();
        auto& rt_info = fc_node->get_rt_info();
        if (rt_info.count("parallelDomain")) {
            return false;
        }

        auto out_shape = fc_node->get_output_partial_shape(0);

        auto src_item = fc_node->get_input_node_shared_ptr(0);
        auto weight_node = fc_node->get_input_node_shared_ptr(1);
        
        std::shared_ptr<Node> wgt_node0, wgt_node1;
        if (auto wgt_node = ov::as_type_ptr<ov::opset12::Constant>(weight_node)) {
            // get input
            // auto src_item = fc_node->get_input_node_shared_ptr(0);
            auto wgt_item = fc_node->get_input_node_shared_ptr(1);
            bool weight_is_dynamic = wgt_item->is_dynamic();
            if (weight_is_dynamic) {
                return false;
            }

            // split weight
            constexpr size_t split_dim = 0; // split happens on the first dimension.
            constexpr size_t split_num = 2; // split the tensor into two parts.
            auto split_dim_range = wgt_item->get_shape()[split_dim];

            // needn't to split fc when the dim is 0.
            if (split_dim_range <= 1) {
                return false;
            }

            auto split_dim_node = std::make_shared<ov::opset8::Constant>(ov::element::i64, ov::Shape{}, split_dim);

            // We should use Split for even dim and VariadicSplit for odd dim.
            std::shared_ptr<Node> split_wgts;
            if (split_dim_range % 2 == 0) {
                split_wgts = std::make_shared<ov::opset12::Split>(wgt_item,
                                                                  split_dim_node,
                                                                  split_num);
            } else {
                auto fc0_dim0 = static_cast<int64_t>((split_dim_range + 1) / 2);
                auto fc1_dim0 = static_cast<int64_t>((split_dim_range - 1) / 2);
                auto split_length = ov::opset8::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {fc0_dim0, fc1_dim0});
                split_wgts = std::make_shared<ov::opset12::VariadicSplit>(wgt_item,
                                                                          split_dim_node,
                                                                          split_length);
            }

            // wgt_node0 = split_wgts->output(0);
            // wgt_node1 = split_wgts->output(1);
            // sub fc
            auto fc_output_type = fc_node->get_output_element_type(0);
            const auto out_rank = out_shape.rank();
            auto fc_node0 = std::make_shared<ov::intel_cpu::FullyConnectedNode>(src_item,
                                                                                split_wgts->output(0),
                                                                                out_rank,
                                                                                fc_output_type);
            auto fc_node1 = std::make_shared<ov::intel_cpu::FullyConnectedNode>(src_item,
                                                                                split_wgts->output(1),
                                                                                out_rank,
                                                                                fc_output_type);
            // runtime parallel
            fc_node0->get_rt_info()["parallelDomain"] = fc_node->get_friendly_name();
            fc_node1->get_rt_info()["parallelDomain"] = fc_node->get_friendly_name();
            // concat
            ov::OutputVector concat_args({fc_node0, fc_node1});
            constexpr size_t concat_dim = -1; // concat happens on the lastest dimension.
            auto concat_node = std::make_shared<ov::opset12::Concat>(concat_args, concat_dim);
            auto concat_shape = concat_node->get_output_partial_shape(0);
            if (concat_shape != out_shape) {
                return false;
            }
            replace_node(fc_node, concat_node);
            return true;
        } else if (auto multiply_node = ov::as_type_ptr<ov::opset12::Multiply>(weight_node)) {
            if (!(ov::as_type_ptr<ov::opset12::Multiply>(multiply_node))) {
                return false;
            }
            auto multiply_pattern = multiply_node->get_input_node_shared_ptr(1);
            if (!multiply_pattern) {
                return false;
            }
            auto subtract_node = multiply_node->get_input_node_shared_ptr(0);
            if (!(ov::as_type_ptr<ov::opset12::Subtract>(subtract_node))) {
                return false;
            }
            auto convert_node1 = subtract_node->get_input_node_shared_ptr(1);
            if (!(ov::as_type_ptr<ov::opset12::Convert>(convert_node1))) {
                return false;
            }
            auto convert_node1_const = convert_node1->get_input_node_shared_ptr(0);
            if (!(ov::as_type_ptr<ov::opset12::Constant>(convert_node1_const))) {
                return false;
            }
            auto convert_node0 = subtract_node->get_input_node_shared_ptr(0);
            if (!(ov::as_type_ptr<ov::opset12::Convert>(convert_node0))) {
                return false;
            }
            auto wgt_item = convert_node0->get_input_node_shared_ptr(0);
            auto cvt_prec = convert_node0->get_element_type();

            constexpr size_t split_dim = 0; // split happens on the first dimension.
            constexpr size_t split_num = 2; // split the tensor into two parts.
            auto split_dim_range = wgt_item->get_shape()[split_dim];
            auto convert_dim_range = convert_node1->get_shape()[split_dim];
            bool need_to_split_convert = split_dim_range == convert_dim_range;

            // needn't to split fc when the dim is 0.
            if (split_dim_range <= 1) {
                return false;
            }

            auto split_dim_node = std::make_shared<ov::opset8::Constant>(ov::element::i64, ov::Shape{}, split_dim);

            // We should use Split for even dim and VariadicSplit for odd dim.
            std::shared_ptr<Node> split_wgts;
            std::shared_ptr<Node> split_muls;
            std::shared_ptr<Node> split_cvts;
            if (split_dim_range % 2 == 0) {
                split_wgts = std::make_shared<ov::opset12::Split>(wgt_item,
                                                                  split_dim_node,
                                                                  split_num);
                split_muls = std::make_shared<ov::opset12::Split>(multiply_pattern,
                                                                  split_dim_node,
                                                                  split_num);
                if (need_to_split_convert) {
                    split_cvts = std::make_shared<ov::opset12::Split>(convert_node1_const,
                                                                      split_dim_node,
                                                                      split_num);
                }
            } else {
                auto fc0_dim0 = static_cast<int64_t>((split_dim_range + 1) / 2);
                auto fc1_dim0 = static_cast<int64_t>((split_dim_range - 1) / 2);
                auto split_length = ov::opset8::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {fc0_dim0, fc1_dim0});
                split_wgts = std::make_shared<ov::opset12::VariadicSplit>(wgt_item,
                                                                          split_dim_node,
                                                                          split_length);
                split_muls = std::make_shared<ov::opset12::VariadicSplit>(multiply_pattern,
                                                                          split_dim_node,
                                                                          split_length);
                if (need_to_split_convert) {
                    split_cvts = std::make_shared<ov::opset12::VariadicSplit>(convert_node1_const,
                                                                              split_dim_node,
                                                                              split_length);
                }
            } 

            auto cvt_node0 = std::make_shared<ov::opset12::Convert>(split_wgts->output(0), cvt_prec);
            std::shared_ptr<Node> sub_node0;
            if (need_to_split_convert) {
                auto convert_split0 = std::make_shared<ov::opset12::Convert>(split_cvts->output(0), cvt_prec);
                ov::pass::disable_constant_folding(convert_split0);
                sub_node0 = std::make_shared<ov::opset12::Subtract>(cvt_node0, convert_split0);
            } else {
                sub_node0 = std::make_shared<ov::opset12::Subtract>(cvt_node0, convert_node1);
            }
            auto mul_node0 = std::make_shared<ov::opset12::Multiply>(sub_node0, split_muls->output(0));

            auto cvt_node1 = std::make_shared<ov::opset12::Convert>(split_wgts->output(1), cvt_prec);
            std::shared_ptr<Node> sub_node1;
            if (need_to_split_convert) {
                auto convert_split1 = std::make_shared<ov::opset12::Convert>(split_cvts->output(1), cvt_prec);
                ov::pass::disable_constant_folding(convert_split1);
                sub_node1 = std::make_shared<ov::opset12::Subtract>(cvt_node1, convert_split1);
            } else {
                sub_node1 = std::make_shared<ov::opset12::Subtract>(cvt_node1, convert_node1);
            }
            auto mul_node1 = std::make_shared<ov::opset12::Multiply>(sub_node1, split_muls->output(1));

            // sub fc
            auto fc_output_type = fc_node->get_output_element_type(0);
            const auto out_rank = out_shape.rank();
            auto fc_node0 = std::make_shared<ov::intel_cpu::FullyConnectedNode>(src_item,
                                                                                mul_node0,
                                                                                out_rank,
                                                                                fc_output_type);
            auto fc_node1 = std::make_shared<ov::intel_cpu::FullyConnectedNode>(src_item,
                                                                                mul_node1,
                                                                                out_rank,
                                                                                fc_output_type);
            // runtime parallel
            fc_node0->get_rt_info()["parallelDomain"] = fc_node->get_friendly_name();
            fc_node1->get_rt_info()["parallelDomain"] = fc_node->get_friendly_name();
            // concat
            ov::OutputVector concat_args({fc_node0, fc_node1});
            constexpr size_t concat_dim = -1; // concat happens on the lastest dimension.
            auto concat_node = std::make_shared<ov::opset12::Concat>(concat_args, concat_dim);
            auto concat_shape = concat_node->get_output_partial_shape(0);
            if (concat_shape != out_shape) {
                return false;
            }
            replace_node(fc_node, concat_node);
            return true;
        } else if (auto reshape_node = ov::as_type_ptr<ov::opset12::Reshape>(weight_node)) {
            auto reshape_pattern = reshape_node->get_input_node_shared_ptr(1);
            auto reshape_const = std::dynamic_pointer_cast<ov::opset12::Constant>(reshape_pattern);
            if (!reshape_pattern || !reshape_const) {
                return false;
            }
            auto rsp_shape = reshape_pattern->get_output_shape(0).to_string();
            auto reshape_vec = reshape_const->get_vector<int32_t>();
            bool reshape_special_zero = reshape_node->get_special_zero();

            auto multiply_node = reshape_node->get_input_node_shared_ptr(0);
            if (!(ov::as_type_ptr<ov::opset12::Multiply>(multiply_node))) {
                return false;
            }
            auto multiply_pattern = multiply_node->get_input_node_shared_ptr(1);
            if (!multiply_pattern) {
                return false;
            }
            auto mul_shape = multiply_pattern->get_shape().to_string();
            auto subtract_node = multiply_node->get_input_node_shared_ptr(0);
            if (!(ov::as_type_ptr<ov::opset12::Subtract>(subtract_node))) {
                return false;
            }
            auto convert_node1 = subtract_node->get_input_node_shared_ptr(1);
            if (!(ov::as_type_ptr<ov::opset12::Convert>(convert_node1))) {
                return false;
            }
            auto convert_node1_const = ov::as_type_ptr<ov::opset12::Constant>(convert_node1->get_input_node_shared_ptr(0));
            // if (!(ov::as_type_ptr<ov::opset12::Constant>(convert_node1_const))) {
            if (convert_node1_const) {
                return false;
            }
            auto convert_node0 = subtract_node->get_input_node_shared_ptr(0);
            if (!(ov::as_type_ptr<ov::opset12::Convert>(convert_node0))) {
                return false;
            }
            auto wgt_item = convert_node0->get_input_node_shared_ptr(0);
            auto cvt_prec = convert_node0->get_element_type();

            constexpr size_t split_dim = 0; // split happens on the first dimension.
            constexpr size_t split_num = 2; // split the tensor into two parts.
            auto split_dim_range = wgt_item->get_shape()[split_dim];
            auto convert_dim_range = convert_node1->get_shape()[split_dim];
            bool need_to_split_convert = split_dim_range == convert_dim_range;
            std::shared_ptr<Node> zp_const0, zp_const1;
            if (!need_to_split_convert) {
                zp_const0 = std::make_shared<ov::opset12::Constant>(convert_node1_const->get_element_type(), convert_node1_const->get_shape(), convert_node1_const->get_data_ptr());
                zp_const1 = std::make_shared<ov::opset12::Constant>(convert_node1_const->get_element_type(), convert_node1_const->get_shape(), convert_node1_const->get_data_ptr());
            }

            // needn't to split fc when the dim is 0.
            if (split_dim_range <= 1) {
                return false;
            }

            auto split_dim_node = std::make_shared<ov::opset8::Constant>(ov::element::i64, ov::Shape{}, split_dim);

            // We should use Split for even dim and VariadicSplit for odd dim.
            std::shared_ptr<Node> split_wgts;
            std::shared_ptr<Node> split_muls;
            std::shared_ptr<Node> split_cvts;
            std::vector<int32_t> split_rsp0, split_rsp1;
            if (split_dim_range % 2 == 0) {
                split_wgts = std::make_shared<ov::opset12::Split>(wgt_item,
                                                                  split_dim_node,
                                                                  split_num);
                split_muls = std::make_shared<ov::opset12::Split>(multiply_pattern,
                                                                  split_dim_node,
                                                                  split_num);
                
                if (need_to_split_convert) {
                    split_cvts = std::make_shared<ov::opset12::Split>(convert_node1_const,
                                                                      split_dim_node,
                                                                      split_num);
                }
                split_rsp0 = {reshape_vec[0]/2, reshape_vec[1]};
                split_rsp1 = {reshape_vec[0]/2, reshape_vec[1]};
            } else {
                auto fc0_dim0 = static_cast<int32_t>((split_dim_range + 1) / 2);
                auto fc1_dim0 = static_cast<int32_t>((split_dim_range - 1) / 2);
                auto split_length = ov::opset8::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {fc0_dim0, fc1_dim0});
                split_wgts = std::make_shared<ov::opset12::VariadicSplit>(wgt_item,
                                                                          split_dim_node,
                                                                          split_length);
                split_muls = std::make_shared<ov::opset12::VariadicSplit>(multiply_pattern,
                                                                          split_dim_node,
                                                                          split_length);
                if (need_to_split_convert) {
                    split_cvts = std::make_shared<ov::opset12::VariadicSplit>(convert_node1_const,
                                                                              split_dim_node,
                                                                              split_length);
                }
                split_rsp0 = {fc0_dim0, reshape_vec[1]};
                split_rsp1 = {fc1_dim0, reshape_vec[1]};
            } 

            auto cvt_node0 = std::make_shared<ov::opset12::Convert>(split_wgts->output(0), cvt_prec);
            // auto sub_node0 = std::make_shared<ov::opset12::Subtract>(cvt_node0, cvt_reuse);
            std::shared_ptr<Node> sub_node0, convert_split0;
            if (need_to_split_convert) {
                convert_split0 = std::make_shared<ov::opset12::Convert>(split_cvts->output(0), cvt_prec);
                sub_node0 = std::make_shared<ov::opset12::Subtract>(cvt_node0, convert_split0);
            } else {
                convert_split0 = std::make_shared<ov::opset12::Convert>(zp_const0, cvt_prec);
                ov::pass::disable_constant_folding(convert_split0);
                sub_node0 = std::make_shared<ov::opset12::Subtract>(cvt_node0, convert_split0);
            }
            auto mul0 = split_muls->output(0);
            auto mul0_shape = mul0.get_node_shared_ptr()->get_output_shape(0).to_string();
            auto mul_node0 = std::make_shared<ov::opset12::Multiply>(sub_node0, split_muls->output(0));

            auto rsp_pattern0 = ov::opset12::Constant::create<int32_t>(ov::element::i64, ov::Shape{2}, split_rsp0);;
            auto rsp_node0 = std::make_shared<ov::opset12::Reshape>(mul_node0, rsp_pattern0, reshape_special_zero);

            auto cvt_node1 = std::make_shared<ov::opset12::Convert>(split_wgts->output(1), cvt_prec);
            // auto sub_node1 = std::make_shared<ov::opset12::Subtract>(cvt_node1, cvt_reuse);
            std::shared_ptr<Node> sub_node1, convert_split1;
            if (need_to_split_convert) {
                convert_split1 = std::make_shared<ov::opset12::Convert>(split_cvts->output(1), cvt_prec);
                sub_node1 = std::make_shared<ov::opset12::Subtract>(cvt_node1, convert_split1);
            } else {
                convert_split1 = std::make_shared<ov::opset12::Convert>(zp_const1, cvt_prec);
                ov::pass::disable_constant_folding(convert_split1);
                sub_node1 = std::make_shared<ov::opset12::Subtract>(cvt_node1, convert_split1);
            }
            auto mul_node1 = std::make_shared<ov::opset12::Multiply>(sub_node1, split_muls->output(1));
            auto rsp_pattern1 = ov::opset12::Constant::create<int32_t>(ov::element::i64, ov::Shape{2}, split_rsp1);;
            auto rsp_node1 = std::make_shared<ov::opset12::Reshape>(mul_node1, rsp_pattern1, reshape_special_zero);

            // sub fc
            auto fc_output_type = fc_node->get_output_element_type(0);
            const auto out_rank = out_shape.rank();
            auto fc_node0 = std::make_shared<ov::intel_cpu::FullyConnectedNode>(src_item,
                                                                                rsp_node0,
                                                                                out_rank,
                                                                                fc_output_type);
            auto fc_node1 = std::make_shared<ov::intel_cpu::FullyConnectedNode>(src_item,
                                                                                rsp_node1,
                                                                                out_rank,
                                                                                fc_output_type);
            // runtime parallel
            fc_node0->get_rt_info()["parallelDomain"] = fc_node->get_friendly_name();
            fc_node1->get_rt_info()["parallelDomain"] = fc_node->get_friendly_name();
            // concat
            ov::OutputVector concat_args({fc_node0, fc_node1});
            constexpr size_t concat_dim = -1; // concat happens on the lastest dimension.
            auto concat_node = std::make_shared<ov::opset12::Concat>(concat_args, concat_dim);
            auto concat_shape = concat_node->get_output_partial_shape(0);
            if (concat_shape != out_shape) {
                return false;
            }
            replace_node(fc_node, concat_node);
            return true;
        } else {
            return false;
        }
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_m, matcher_name);
    this->register_matcher(m, callback);
}
