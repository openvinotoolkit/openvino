// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_split_decomposition.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include "openvino/opsets/opset1.hpp"
#include <openvino/opsets/opset13.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include "openvino/op/gather.hpp"

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;
using namespace ov;

void pass::MatmulSplitDecomposition::split_weights(const Output<Node>& weights, OutputVector& new_weights,
                                                   const Output<Node>& bias, OutputVector& new_bias) {
    const auto& weights_shape = weights.get_partial_shape();
    int64_t weights_rank = static_cast<int64_t>(weights_shape.rank().get_length());

    const auto& bias_shape = bias.get_partial_shape();
    int64_t bias_rank = static_cast<int64_t>(bias_shape.rank().get_length());

    if (weights_rank != 2 || bias_rank != 3) {
        return;
    }

    auto axis = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto split = register_new_node<opset1::Split>(weights, axis, 3);

    // Constantfold new weights
    for (auto& out : split->outputs()) {
        if (auto constant = ov::util::get_constant_from_source(out)) {  // TODO: why Convert cannot be constfolded?
            new_weights.emplace_back(constant->shared_from_this());
        } else
            new_weights.emplace_back(out);
    }

    auto axis2 = register_new_node(v0::Constant::create(element::i32, Shape{}, {2}));
    auto split2 = register_new_node<opset1::Split>(bias, axis2, 3);

    // Constantfold new bias
    for (auto& out : split2->outputs()) {
        if (auto constant = ov::util::get_constant_from_source(out)) {
            new_bias.emplace_back(constant->shared_from_this());
        } else
            new_bias.emplace_back(out);
    }
}

pass::MatmulSplitDecomposition::MatmulSplitDecomposition() {
    MATCHER_SCOPE(MatmulSplitDecomposition);
    using namespace ov::pass::pattern;

    auto check_zero = [] (Output<Node> output) -> bool {
        auto node = std::dynamic_pointer_cast<opset1::Constant>(output.get_node_shared_ptr());
        const auto& bcst_arg = node->cast_vector<float>();
        return std::all_of(bcst_arg.begin(), bcst_arg.end(), [](float i) {
            return i == 0.0f;
        });
    };

    auto input_pattern = any_input();
    auto matmul_pattern = wrap_type<opset1::MatMul>({input_pattern, any_input()});  // TODO: input2 rank 2

    auto bias_pattern = wrap_type<opset1::Constant>();
    auto add_pattern = wrap_type<opset1::Add>({matmul_pattern, bias_pattern});

    auto reshape_pattern = wrap_type<opset1::Reshape>({add_pattern, any_input()});  // TODO: input2 shape [5]
    auto transpose_pattern = wrap_type<opset6::Transpose>({reshape_pattern, any_input()});  // TODO: input2 shape [5]
    
    auto constant_indices0 = wrap_type<opset1::Constant>(check_zero);
    auto constant_axis0 = wrap_type<opset1::Constant>(check_zero);
    auto gather0_pattern = wrap_type<ov::op::util::GatherBase>({transpose_pattern, wrap_type<v0::Constant>(), wrap_type<v0::Constant>()});

    std::cout << "==================1=============\n\n";

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& transpose = pattern_map.at(transpose_pattern).get_node_shared_ptr();
        std::cout << "==================2============="<< transpose->get_friendly_name() << "\n\n";

        auto matmul = pattern_map.at(matmul_pattern).get_node_shared_ptr();
        auto weights = matmul->input_value(1);
        auto add = pattern_map.at(add_pattern).get_node_shared_ptr();
        const auto& bias = pattern_map.at(bias_pattern);

        const auto& reshape = pattern_map.at(reshape_pattern);
        auto concat = reshape.get_node_shared_ptr()->input_value(1);

        // there should be only 3 gathers to split transpose
        auto children = transpose->get_output_target_inputs(0);
        NodeVector gathers;
        gathers.resize(3);
        std::cout << "==================3=============\n\n";
        for (auto& child : children) {
            auto gather = child.get_node()->shared_from_this();
            if (ov::is_type<ov::op::util::GatherBase>(gather)) {
                std::cout << "==================3.1=============\n\n";
                const auto axis_node = as_type_ptr<opset6::Constant>(gather->input_value(2).get_node_shared_ptr());
                const auto& axis_val = axis_node->cast_vector<int32_t>();
                if (axis_val.size() != 1) return false;
                std::cout << "==================3.2=============\n\n";
                if (axis_val[0] != 0) return false;

                std::cout << "==================3.3=============\n\n";
                const auto indices_node = as_type_ptr<opset6::Constant>(gather->input_value(1).get_node_shared_ptr());
                const auto& indices_val = indices_node->cast_vector<int32_t>();
                if (indices_val.size() != 1) return false;
                std::cout << "==================3.4=============" << indices_val[0] << "\n\n";
                if (indices_val[0] < 0 || indices_val[0] >= 3) return false;

                std::cout << "==================3.4=============\n\n";
                
                gathers[indices_val[0]] = gather;
            }
        }
        std::cout << "==================4=============\n\n";
        if (std::any_of(gathers.begin(), gathers.end(), [](const std::shared_ptr<Node> node_ptr) {
                        return !node_ptr || !is_type<ov::op::util::GatherBase>(node_ptr);
                    })) return false;

        OutputVector new_weights, new_bias;
        split_weights(weights, new_weights, bias, new_bias);
        if (new_weights.size() != 3 || new_bias.size() != 3)
            return false;
        std::cout << "==================5=============\n\n";

        auto const_indices = register_new_node(v0::Constant::create(element::i32, Shape{4}, {0, 1, 3, 4}));
        auto const_axis = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));
        auto new_shape = register_new_node<v1::Gather>(concat, const_indices, const_axis);

        const auto& input = pattern_map.at(input_pattern);

        for (size_t i = 0 ; i < 3; i++) {
            std::cout << "=========replace=========5." << i << "=============\n\n";
            auto mm0 = register_new_node<v0::MatMul>(input, new_weights[i]);
            auto add0 = register_new_node<v1::Add>(mm0, new_bias[i]);
            auto reshape0 = register_new_node<v1::Reshape>(add0, new_shape, true);
            auto transpose_order = register_new_node(v0::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3}));
            auto transpose0 = register_new_node<v1::Transpose>(reshape0, transpose_order);
            
            copy_runtime_info({gathers[i], weights.get_node_shared_ptr(), bias.get_node_shared_ptr(), matmul, add, transpose}, get_new_nodes());
            replace_node(gathers[i], transpose0);  // replace gatherX by transposeX

            std::cout << "==================5." << i << "=============\n\n";
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_pattern, matcher_name);
    this->register_matcher(m, callback);
}
