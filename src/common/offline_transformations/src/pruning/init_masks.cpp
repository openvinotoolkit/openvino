// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mask_attribute.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"
#include "pruning.hpp"

namespace ov {
namespace pass {
namespace init_masks {

class InitConvMask;
class InitMatMulMask;

}  // namespace init_masks
}  // namespace pass
}  // namespace ov

class ov::pass::init_masks::InitConvMask : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("init_masks::InitConvMask");
    InitConvMask() {
        auto input = pattern::any_input();
        auto weights = pattern::any_input();
        auto conv = pattern::wrap_type<opset6::Convolution, opset6::GroupConvolution>({input, weights});

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& m_output = pattern_map.at(conv);

            // Initializing weights mask:
            // 1. Looking for Const node with weights
            NodeVector weights_calculation_nodes;
            auto cur_node = m_output.get_node()->get_input_node_shared_ptr(1);

            while (!ov::is_type<opset6::Constant>(cur_node) && cur_node->inputs().size()) {
                weights_calculation_nodes.push_back(cur_node);
                cur_node = cur_node->get_input_node_shared_ptr(0);
            }
            if (!ov::is_type<opset6::Constant>(cur_node)) {
                OPENVINO_DEBUG("Can't find Constant weights for Convolution: ",
                               m_output.get_node()->get_friendly_name(),
                               "\n");
                return false;
            }

            // 2. Init mask for Const node
            InitConstMask({0} /* check only output channels dim */).apply(cur_node);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(conv, "ConvolutionInitMask");
        register_matcher(m, callback);
    }
};

class ov::pass::init_masks::InitMatMulMask : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("init_masks::InitMatMulMask");
    InitMatMulMask() {
        auto a = pattern::any_input();
        auto b = pattern::any_input();
        auto matmul_pattern = pattern::wrap_type<opset6::MatMul>({a, b});

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& matmul = ov::as_type_ptr<opset6::MatMul>(pattern_map.at(matmul_pattern).get_node_shared_ptr());
            if (!matmul)
                return false;

            // Assume constant always in the first input port.
            // Initializing weights mask:
            // 1. Looking for Const node with weights
            NodeVector weights_calculation_nodes;
            auto cur_node = matmul->get_input_node_shared_ptr(1);

            if (cur_node->get_output_partial_shape(0).is_dynamic())
                return false;
            const auto input_size = cur_node->get_output_shape(0).size();
            auto dim_order = std::vector<int64_t>(input_size);
            std::iota(dim_order.begin(), dim_order.end(), 0);

            while (!ov::is_type<opset6::Constant>(cur_node) && cur_node->inputs().size()) {
                weights_calculation_nodes.push_back(cur_node);
                if (ov::is_type<opset6::Transpose>(cur_node)) {
                    const auto forward_order =
                        ov::util::get_constant_from_source(cur_node->get_input_node_shared_ptr(1));
                    if (!forward_order)
                        return false;
                    const auto forward_order_vec = forward_order->cast_vector<int64_t>();
                    if (forward_order_vec.size() != input_size)
                        return false;
                    auto new_order = std::vector<int64_t>(forward_order_vec.size());
                    for (size_t i = 0; i < forward_order_vec.size(); ++i) {
                        new_order[forward_order_vec[i]] = dim_order[i];
                    }
                    dim_order = new_order;
                } else {
                    if (ov::is_type<opset6::Reshape>(cur_node) || ov::is_type<opset6::MatMul>(cur_node)) {
                        OPENVINO_DEBUG("Can't init mask for MatMul: ",
                                       matmul->get_friendly_name(),
                                       " because of node ",
                                       cur_node->get_friendly_name(),
                                       " in the way from weights to Matmul\n");
                        return false;
                    }
                }
                cur_node = cur_node->get_input_node_shared_ptr(0);
            }
            if (!ov::is_type<opset6::Constant>(cur_node)) {
                OPENVINO_DEBUG("Can't find Constant weights for MatMul: ", matmul->get_friendly_name(), "\n");
                return false;
            }
            // 2. Get constant rank to set mask on last dimension
            const auto const_op = ov::as_type_ptr<opset6::Constant>(cur_node);
            const auto shape_rank = const_op->get_shape().size();
            const size_t shift = (matmul->get_transpose_b()) ? 2 : 1;
            if (shape_rank < shift) {
                OPENVINO_DEBUG("Can't init mask for MatMul: ", matmul->get_friendly_name(), "\n");
                return false;
            }
            const auto idx = shape_rank - shift;
            const size_t outer_dim = std::find(dim_order.begin(), dim_order.end(), idx) - dim_order.begin();
            // 3. Init mask for Const node
            InitConstMask({outer_dim} /* check only outer dim */).apply(cur_node);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_pattern, "MatMulInitMask");
        register_matcher(m, callback);
    }
};

ov::pass::InitMasks::InitMasks() {
    add_matcher<init_masks::InitConvMask>();
    add_matcher<init_masks::InitMatMulMask>();
}
