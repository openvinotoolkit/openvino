// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "split_fc.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
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

        // get input
        auto src_item = fc_node->get_input_node_shared_ptr(0);
        auto wgt_item = fc_node->get_input_node_shared_ptr(1);
        bool weight_is_dynamic = wgt_item->is_dynamic();
        if (weight_is_dynamic) {
            return false;
        }

        auto out_shape = fc_node->get_output_partial_shape(0);

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
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_m, matcher_name);
    this->register_matcher(m, callback);
}
