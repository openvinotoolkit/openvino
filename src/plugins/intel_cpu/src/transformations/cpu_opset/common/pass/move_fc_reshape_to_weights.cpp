// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "move_fc_reshape_to_weights.hpp"
#include <transformations/utils/utils.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/opsets/opset1.hpp>

#include "itt.hpp"

ov::intel_cpu::MoveFCReshapeToWeights::MoveFCReshapeToWeights() {
    MATCHER_SCOPE(MoveFCReshapeToWeights);
    using namespace ov::pass::pattern;
    auto weights_m = wrap_type<ov::opset1::Constant>(consumers_count(1));
    auto convert_m = wrap_type<ov::opset1::Convert>({weights_m});

    auto sub_const_m = wrap_type<ov::opset1::Constant>(consumers_count(1));
    auto subtract_m = wrap_type<ov::opset1::Subtract>({convert_m, sub_const_m});

    auto mul_const_m = wrap_type<ov::opset1::Constant>(consumers_count(1));
    auto mul_with_sub_m = wrap_type<ov::opset1::Multiply>({subtract_m, mul_const_m}, rank_equals(3));
    auto mul_no_sub_m = wrap_type<ov::opset1::Multiply>({convert_m, mul_const_m}, rank_equals(3));
    auto mul_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{mul_with_sub_m, mul_no_sub_m});

    auto one_consumer_rank_2 = [](const ov::Output<ov::Node>& out) {
        return consumers_count(1)(out) && rank_equals(2)(out);
    };
    auto reshape_const_m = wrap_type<ov::opset1::Constant>();
    auto reshape_m = wrap_type<ov::opset1::Reshape>({mul_m, reshape_const_m}, one_consumer_rank_2);

    auto transpose_const_m = wrap_type<ov::opset1::Constant>();
    auto transpose_m = wrap_type<ov::opset1::Transpose>({reshape_m, transpose_const_m});
    auto weights_input_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{reshape_m, transpose_m});

    auto data_m = any_input();
    auto fully_connected_m = wrap_type<ov::intel_cpu::FullyConnectedNode>({data_m, weights_input_m});

    ov::matcher_pass_callback callback = [&](ov::pass::pattern::Matcher& m) {
        const auto fully_connected = m.get_match_root();
        const auto weights_path = fully_connected->get_input_node_shared_ptr(1);
        const bool with_transpose = ov::is_type<ov::opset1::Transpose>(weights_path);
        if (with_transpose) {
            const auto transpose_const = ov::as_type_ptr<ov::opset1::Constant>(weights_path->get_input_node_shared_ptr(1));
            if (transpose_const->cast_vector<int>() != std::vector<int>{1, 0}) {
                return false;
            }
        }

        const auto reshape = with_transpose ? weights_path->get_input_node_shared_ptr(0) : weights_path;
        std::cout << "All is OK\n\n" << std::endl;
        // // Create FullyConnected
        // auto output_rank = matmul->get_output_partial_shape(0).rank();
        // auto fc = std::make_shared<ov::intel_cpu::FullyConnectedNode>(fc_input_a, fc_input_b, output_rank,
        //         matmul->get_output_element_type(0));
        // fc->set_friendly_name(matmul->get_friendly_name());
        // new_ops.push_back(fc);
        // ov::copy_runtime_info(matmul, new_ops);
        // ov::replace_node(matmul, fc);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m, matcher_name);
    this->register_matcher(m, callback);
}
