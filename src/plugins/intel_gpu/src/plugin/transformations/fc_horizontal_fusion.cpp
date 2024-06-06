// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_horizontal_fusion.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

#include "openvino/core/rt_info.hpp"
#include <openvino/opsets/opset1.hpp>
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "intel_gpu/op/placeholder.hpp"

namespace ov {
namespace intel_gpu {

FullyConnectedHorizontalFusion::FullyConnectedHorizontalFusion() {
    using namespace ov::pass::pattern;

    auto input = any_input(consumers_more_than(3));
    auto weight1 = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto weight2 = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto weight3 = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto bias1 = any_input();
    auto bias2 = any_input();
    auto bias3 = any_input();
    auto scale1 = any_input();
    auto scale2 = any_input();
    auto scale3 = any_input();
 
    auto fc1 = wrap_type<op::FullyConnectedCompressed>({input, weight1, bias1, scale1, any_input()}, consumers_count(1));
    auto fc2 = wrap_type<op::FullyConnectedCompressed>({input, weight2, bias2, scale2, any_input()}, consumers_count(1));
    auto fc3 = wrap_type<op::FullyConnectedCompressed>({input, weight3, bias3, scale3, any_input()}, consumers_count(1));

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& m_input = pattern_map.at(input).get_node_shared_ptr();
        std::shared_ptr<Node> m_fc = pattern_map.at(fc3).get_node_shared_ptr();
        auto m_bias = pattern_map.at(bias3).get_node_shared_ptr();
        // bias is not supported yet
        // also only scalar zp supported
        if (!std::dynamic_pointer_cast<op::Placeholder>(m_bias)) {
            std::cout << "there is bias!" << m_bias->get_friendly_name() << std::endl;
            return false;
        }
        auto input_node = m_fc->get_input_node_shared_ptr(0);
        std::vector<std::shared_ptr<op::FullyConnectedCompressed>> fc_nodes;
        ov::NodeVector weight_nodes;
        ov::NodeVector scale_nodes;
        for (auto user : input_node->get_users()) {
            auto fc_user = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(user);
            if (fc_user) {
                fc_nodes.push_back(fc_user);
                auto weight = fc_user->get_input_node_shared_ptr(1);
                weight_nodes.push_back(weight);
                auto scale = fc_user->get_input_node_shared_ptr(3);
                scale_nodes.push_back(scale);
            }
        }
        if (fc_nodes.size() != 3)
            return false;
        std::cout << "Found target FC nodes to fuse: " << std::endl;
        auto weight_dtype = fc_nodes[0]->get_input_element_type(1);
        auto zp_node = fc_nodes[0]->get_input_node_shared_ptr(4);
        std::vector<int64_t> out_n_sizes;
        auto new_n_size = 0;
        auto k_size = fc_nodes[0]->get_input_shape(1)[fc_nodes[0]->get_input_shape(1).size() - 1];
        // merge weights, scale, zp
        for (auto fc : fc_nodes) {
            if (k_size != fc->get_input_shape(1)[fc->get_input_shape(1).size() - 1])
                return false;
            if (weight_dtype != fc->get_input_element_type(1))
                return false;
            new_n_size += fc->get_input_shape(1)[fc->get_input_shape(1).size() - 2];
            out_n_sizes.push_back(fc->get_input_shape(1)[fc->get_input_shape(1).size() - 2]);
            std::cout << " === " << fc->get_friendly_name() << " " << fc->get_input_shape(1).to_string() << std::endl;
            std::cout << "     " << "has " << fc->get_input_size() << " inputs" << std::endl;
            for (size_t i = 0; i < fc->get_input_size(); ++i) {
                std::cout << "      input[" << i << "] : " << fc->get_input_node_shared_ptr(i)->get_friendly_name() << " ("
                          << fc->get_input_partial_shape(i).to_string() << ") " << std::endl;
                if (i == fc->get_input_size() - 1) {
                    // zeropoint
                    auto zp_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(fc->get_input_node_shared_ptr(i));
                    if (zp_node) {
                        auto zp_val = zp_node->cast_vector<int32_t>();
                        std::cout << "           zp[0] value : " << zp_val[0] << std::endl;
                    }
                }
            }
        }
        std::cout << " === dtype : " << weight_dtype << std::endl;
        auto weight_nodes_as_output_vector = ov::OutputVector{weight_nodes[0], weight_nodes[1], weight_nodes[2]};
        auto fused_weight = std::make_shared<ov::op::v0::Concat>(weight_nodes_as_output_vector, 0);
        auto scale_nodes_as_output_vector = ov::OutputVector{scale_nodes[0], scale_nodes[1], scale_nodes[2]};
        auto fused_scale = std::make_shared<ov::op::v0::Concat>(scale_nodes_as_output_vector, 0);
        auto new_fc = std::make_shared<op::FullyConnectedCompressed>(input_node, fused_weight, m_bias, fused_scale, zp_node);
        auto new_fc_name = fc_nodes[0]->get_friendly_name() + "fused";
        new_fc->set_friendly_name(new_fc_name);
        copy_runtime_info(fc_nodes[0], new_fc);

        std::cout << "=> Fused to new single fc " << new_fc->get_friendly_name() << std::endl;
        std::cout << " === weight new shape : " << new_n_size << ","  << k_size << std::endl;

        auto split_name = fc_nodes[0]->get_friendly_name() + "split";
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {new_fc->get_output_partial_shape(0).size() - 1});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, out_n_sizes);
        auto output_split = std::make_shared<ov::op::v1::VariadicSplit>(new_fc, axis_const, split_const);
        output_split->set_friendly_name(split_name);
        auto out_0 = output_split->output(0);
        auto out_1 = output_split->output(1);
        auto out_2 = output_split->output(2);
        for (size_t i = 0; i < fc_nodes.size(); ++i) {
            // consumers count limit : 1
            auto user_node = fc_nodes[i]->get_users()[0];
            user_node->input(0).replace_source_output(output_split->output(i));
            fc_nodes[i]->clear_control_dependents();
        }

        // add variable split of output

//        const auto& m_data = pattern_map.at(data).get_node_shared_ptr();
//        const auto& m_weights = pattern_map.at(weights).get_node_shared_ptr();
//        const auto& m_bias = pattern_map.at(bias).get_node_shared_ptr();
//        const auto& m_convert = pattern_map.at(convert).get_node_shared_ptr();
//        auto output_type = m_convert->get_output_element_type(0);
//
//        std::shared_ptr<Node> m_fc = nullptr;
//        std::shared_ptr<Node> new_fc = nullptr;
//        auto it = pattern_map.find(fully_connected);
//        if (it != pattern_map.end()) {
//            m_fc = it->second.get_node_shared_ptr();
//            new_fc = std::make_shared<op::FullyConnected>(m_data, m_weights, m_bias, output_type);
//        } else {
//            m_fc = pattern_map.at(fully_connected_compressed).get_node_shared_ptr();
//            new_fc = std::make_shared<op::FullyConnectedCompressed>(m_data,
//                                                                    m_weights,
//                                                                    m_bias,
//                                                                    m_fc->input_value(3),
//                                                                    m_fc->input_value(4),
//                                                                    output_type);
//        }
//        new_fc->set_friendly_name(m_convert->get_friendly_name());
//        copy_runtime_info(m.get_matched_nodes(), new_fc);
//        replace_node(m_convert, new_fc);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc3, "FullyConnectedHorizontalFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
