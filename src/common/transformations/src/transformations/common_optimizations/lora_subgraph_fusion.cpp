// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/lora_subgraph_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/lora_subgraph.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::LoraSubgraphFusion::LoraSubgraphFusion() {
    MATCHER_SCOPE(LoraSubgraphFusion);
    using namespace pass::pattern;
    auto lora_input_m = any_input();
    auto transpose_const1_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose1_m = optional<ov::op::v1::Transpose>({lora_input_m, transpose_const1_m}, consumers_count(1));

    auto read_value1_m = wrap_type<ov::op::util::ReadValueBase>();
    auto convert1_m = optional<ov::op::v0::Convert>(read_value1_m, consumers_count(1));
    auto matmul1_m = wrap_type<ov::op::v0::MatMul>({transpose1_m, convert1_m}, consumers_count(1));

    auto read_value2_m = wrap_type<ov::op::util::ReadValueBase>();
    auto convert2_m = optional<ov::op::v0::Convert>(read_value2_m, consumers_count(1));
    auto multiply_m = wrap_type<ov::op::v1::Multiply>({matmul1_m, convert2_m}, consumers_count(1));

    auto read_value3_m = wrap_type<ov::op::util::ReadValueBase>();
    auto convert3_m = optional<ov::op::v0::Convert>(read_value3_m, consumers_count(1));
    auto matmul2_m = wrap_type<ov::op::v0::MatMul>({multiply_m, convert3_m}, consumers_count(1));

    auto transpose_const2_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose2_m = optional<ov::op::v1::Transpose>({matmul2_m, transpose_const2_m}, consumers_count(1));
    auto main_flow_m = wrap_type<ov::op::v0::MatMul, ov::op::v1::Convolution>({lora_input_m, any_input()});
    auto add_m = wrap_type<ov::op::v1::Add>({transpose2_m, main_flow_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& lora_input = pattern_map.at(lora_input_m);
        const auto& matmul1 = pattern_map.at(matmul1_m);
        const auto& state_1 =
            pattern_map.count(convert1_m) ? pattern_map.at(convert1_m) : pattern_map.at(read_value1_m);
        const auto& multiply = pattern_map.at(multiply_m);
        const auto& state_2 =
            pattern_map.count(convert2_m) ? pattern_map.at(convert2_m) : pattern_map.at(read_value2_m);
        const auto& matmul2 = pattern_map.at(matmul2_m);
        const auto& state_3 =
            pattern_map.count(convert3_m) ? pattern_map.at(convert3_m) : pattern_map.at(read_value3_m);
        const auto& main_flow = pattern_map.at(main_flow_m);
        const auto& add = pattern_map.at(add_m);

        const auto add_node = add.get_node_shared_ptr();
        if (transformation_callback(add_node)) {
            return false;
        }

        auto find_connected_input = [](ov::Node* child, ov::Node* parent) {
            for (size_t i = 0; i < child->get_input_size(); ++i) {
                auto input = child->input(i);
                if (input.get_source_output().get_node() == parent)
                    return input;
            }
            OPENVINO_THROW("Ops are not connected");
        };

        // Note: internal_inputs/external_connections order corresponds to LoraSubgraph semantic
        const std::vector<ov::Input<ov::Node>> internal_inputs{
            // For commutative eltwise ops, input idx may be any, so it must be computed
            find_connected_input(add.get_node(), main_flow.get_node()),
            pattern_map.count(transpose1_m) ? pattern_map.at(transpose1_m).get_node()->input(0)
                                            : matmul1.get_node()->input(0),
            matmul1.get_node()->input(1),
            find_connected_input(multiply.get_node(), state_2.get_node()),
            matmul2.get_node()->input(1),
        };
        const ov::OutputVector external_connections{
            main_flow,
            lora_input,
            state_1,
            state_2,
            state_3,
        };

        ov::ParameterVector subgraph_parameters;
        subgraph_parameters.reserve(internal_inputs.size());
        for (auto& in : internal_inputs) {
            auto new_parameter = std::make_shared<ov::op::v0::Parameter>(in.get_element_type(), in.get_partial_shape());
            subgraph_parameters.push_back(new_parameter);
            in.replace_source_output(new_parameter);
        }
        // Note: lora consumers should be taken before lora_subgraph creation,
        // because only original consumers should be replaced with lora's output
        const auto& lora_consumers = add.get_target_inputs();
        const auto lora_subgraph = std::make_shared<ov::Model>(ov::OutputVector{add}, subgraph_parameters);
        const auto lora_node = std::make_shared<ov::op::internal::LoraSubgraph>(external_connections, lora_subgraph);
        ov::copy_runtime_info(m.get_matched_nodes(), lora_node);
        lora_node->set_friendly_name(add_node->get_friendly_name());

        for (const auto& consumer : lora_consumers)
            consumer.replace_source_output(lora_node->output(0));
        if (!add.get_names().empty())
            lora_node->output(0).set_names(add.get_names());
        return true;
    };

    auto m = std::make_shared<Matcher>(add_m, matcher_name);
    this->register_matcher(m, callback);
}
