// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "apply_mha_fusion.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "intel_gpu/op/mha.hpp"

#include "openvino/core/partial_shape.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/constant.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

// Currently supported shape is limited
const std::string MatchingInputSize = "[10,9216,64]";
const std::string MatchingTransPosedInputSize = "[10,64,9216]";
const std::string MatchingPSize = "[10,9216,9216]";  // P = Q*K

ApplyMHAFusion::ApplyMHAFusion() {
    using namespace ov::pass::pattern;

    auto input0 = any_input(has_static_rank());
    // auto input1 = any_input(has_static_rank());

    auto trans_in0 = any_input(has_static_rank());
    auto trans_const0 = wrap_type<ov::op::v0::Constant>();
    auto transpose_pattern = wrap_type<ov::op::v1::Transpose>({trans_in0, trans_const0});
    auto reshape_pattern = wrap_type<ov::op::v1::Reshape>({transpose_pattern, any_input()});
    auto multi_pattern = wrap_type<ov::op::v1::Multiply>({reshape_pattern, any_input()});
    auto matmul_in1 = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{transpose_pattern, reshape_pattern, multi_pattern});

    // auto matmul_pattern0 = wrap_type<ov::op::v0::MatMul>({input0, input1});
    auto matmul_pattern0 = wrap_type<ov::op::v0::MatMul>({input0, matmul_in1});
    auto softmax_pattern = wrap_type<ov::op::v1::Softmax, ov::op::v8::Softmax>({matmul_pattern0});

    auto input2 = any_input(has_static_rank());
    auto matmul_pattern1 = wrap_type<ov::op::v0::MatMul>({softmax_pattern, input2});

    std::cout << ">> Run ApplyMHAFusion() ============" << std::endl;

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        std::cout << ">> callback : ApplyMHAFusion()" << std::endl;
        OPENVINO_ASSERT(pattern_map.count(matmul_pattern0));
        OPENVINO_ASSERT(pattern_map.count(softmax_pattern));
        OPENVINO_ASSERT(pattern_map.count(matmul_pattern1));

        auto matmul0 = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_pattern0).get_node_shared_ptr());
        if (!matmul0 || transformation_callback(matmul0))
            return false;

        auto input_q = pattern_map.at(input0).get_node_shared_ptr();
        // auto input_k = pattern_map.at(input1).get_node_shared_ptr();
        auto input_k = matmul0->get_input_node_shared_ptr(1);
        auto input_v = pattern_map.at(input2).get_node_shared_ptr();

        // Check shape
        auto shape_q = pattern_map.at(input0).get_partial_shape();
        // auto shape_k = pattern_map.at(input1).get_partial_shape();
        auto shape_k = matmul0->get_input_partial_shape(1);
        auto shape_v = pattern_map.at(input2).get_partial_shape();
        auto shape_p = matmul0->get_output_partial_shape(0);
        auto target_input_shape = ov::PartialShape(MatchingInputSize);
        if (shape_q != target_input_shape || shape_k != target_input_shape || shape_v != target_input_shape ||
            shape_p != ov::PartialShape(MatchingPSize))
            return false;

        // Handle Key values
        // auto transpose =
        //     std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_pattern).get_node_shared_ptr());
        // auto transp_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
        // auto transp_order_values = transp_order->get_axis_vector_val();
        // std::vector<size_t> new_transpose_order = transp_order_values;
        // new_transpose_order[2] = transp_order_values[3];
        // new_transpose_order[3] = transp_order_values[2];
        // const auto& new_transpose_const = ov::op::v0::Constant::create(transp_order->get_element_type(),
        //                                                            {new_transpose_order.size()},
        //                                                            new_transpose_order);
        // new_transpose_const->set_friendly_name(transp_order->get_friendly_name());
        // copy_runtime_info(transp_order, new_transpose_const);
        // replace_node(transp_order, new_transpose_const);

        // auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(pattern_map.at(reshape_pattern).get_node_shared_ptr());
        // auto new_reshape = reshape->clone_with_new_inputs({ reshape->get_input_node_shared_ptr(0), reshape->get_input_node_shared_ptr(1) });
        // new_reshape->set_output_type(0, reshape->get_element_type(), ov::PartialShape(MatchingTransPosedInputSize));
        // new_reshape->set_friendly_name(reshape->get_friendly_name());
        // copy_runtime_info(reshape, new_reshape);
        // replace_node(reshape, new_reshape);

        ov::NodeVector new_ops;
        auto create_transpose = [this, &new_ops ](const ov::Output<ov::Node>& node, const std::string& transpose_name) {
            std::vector<size_t> transpose_order(node.get_partial_shape().size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ transpose_order.size() }, transpose_order);
            auto transpose = std::make_shared<ov::op::v1::Transpose>(node, transpose_const);
            if (!ov::is_type<ov::op::v0::Constant>(transpose)) {
                new_ops.push_back(transpose_const);
                MatcherPass::register_new_node(transpose);
            }
            transpose->set_friendly_name(transpose_name);
            ov::disable_constant_folding(transpose);
            new_ops.push_back(transpose);
            return transpose;
        };

        auto new_input_k = create_transpose(input_k, matmul0->get_friendly_name() + "/transpose_k");
        new_input_k->set_output_type(0, matmul0->get_output_element_type(0), new_input_k->get_output_partial_shape(0));
        auto new_shape_k = new_input_k->get_output_partial_shape(0);

        // std::vector<size_t> transpose_order({0,2,1});
        // auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ transpose_order.size() }, transpose_order);
        // auto transpose = std::make_shared<ov::op::v1::Transpose>(input_k, transpose_const);
        // new_ops.push_back(transpose);

        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_pattern1).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul))
            return false;

#if 1
        std::cout << "  -- node : " << matmul0->get_friendly_name() << std::endl;
        for (size_t i = 0 ; i < matmul0->get_input_size(); i++) {
            auto dep = matmul0->get_input_node_ptr(i);
            std::cout << "    -- dep : " << dep->get_friendly_name() << " : "
                     << dep->get_type_name() << std::endl;
            for (size_t ii = 0 ; ii < dep->get_input_size(); ii++) {
                auto dep2 = dep->get_input_node_ptr(ii);
                std::cout << "      -- dep2 : " << dep2->get_friendly_name() << " : "
                     << dep2->get_type_name() << std::endl;
                for (size_t iii = 0 ; iii < dep2->get_input_size(); iii++) {
                    auto dep3 = dep2->get_input_node_ptr(iii);
                    std::cout << "        -- dep3 : " << dep3->get_friendly_name() << " : "
                                << dep3->get_type_name() << std::endl;
                }
            }
        }

        std::cout << "  -- shape q : " << shape_q.to_string() << std::endl;
        std::cout << "  -- shape k : " << new_shape_k.to_string() << std::endl;
        std::cout << "  -- shape v : " << shape_v.to_string() << std::endl;
        std::cout << "  -- output of matmul0 : " << shape_p << std::endl;
        auto tmp = pattern_map.at(matmul_pattern0).get_node_shared_ptr();
        std::cout << "    -- " << tmp->get_friendly_name() << " : " << tmp->get_type_name() << " : "
                     << tmp->get_output_element_type(0).get_type_name() << std::endl;
#endif

        ov::NodeVector nodes_to_copy_info{pattern_map.at(matmul_pattern0).get_node_shared_ptr(),
                                          pattern_map.at(softmax_pattern).get_node_shared_ptr(),
                                          pattern_map.at(matmul_pattern1).get_node_shared_ptr()};

        std::shared_ptr<ov::Node> mha = std::make_shared<op::MhaFusion>(input_q,
                                                                        new_input_k,  // input_k,
                                                                        input_v,
                                                                        matmul->get_output_element_type(0));
        mha->set_friendly_name(matmul->get_friendly_name()+ "/mha");
        for (size_t i=0 ; i < matmul->get_output_size() ; i++) {
            mha->set_output_type(i, matmul->get_output_element_type(i), matmul->get_output_partial_shape(i));
        }
        new_ops.push_back(mha);
        copy_runtime_info(nodes_to_copy_info, new_ops);
        replace_node(matmul, mha);

        return true;
    };

    auto m = std::make_shared<Matcher>(matmul_pattern1);
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
