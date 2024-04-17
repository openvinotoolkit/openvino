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
#include "openvino/op/reshape.hpp"
#include "openvino/op/multiply.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

// Currently supported shape is limited
const char* MatchingInputSize = "[10,9216,64]";                 // Q, V
const char* MatchingTransPosedInputSize = "[10,64,9216]";       // K
const char* MatchingPSize = "[10,9216,9216]";                   // P = Q*K
const char* ReshapeMatchingInputSize = "[1,10,9216,64]";
const char* ReshapeMatchingTransPosedInputSize = "[1,10,64,9216]";
const char* ReshapeMatchingOutputSize = "[1,10,9216,64]";

ApplyMHAFusion::ApplyMHAFusion() {
    using namespace ov::pass::pattern;

    // Input Query
    auto input0 = any_input(has_static_rank());

    // Input key + transpose : MHA primitive does NOT support input-transpose, not like matmul
    auto trans_in0 = any_input(has_static_rank());
    auto trans_const0 = wrap_type<ov::op::v0::Constant>();
    auto transpose_pattern = wrap_type<ov::op::v1::Transpose>({trans_in0, trans_const0});
    auto reshape_pattern = wrap_type<ov::op::v1::Reshape>({transpose_pattern, any_input()});
    auto multi_pattern = wrap_type<ov::op::v1::Multiply>({reshape_pattern, any_input()});
    auto matmul_in1 = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{transpose_pattern, reshape_pattern, multi_pattern});

    auto matmul_pattern0 = wrap_type<ov::op::v0::MatMul>({input0, matmul_in1});
    auto softmax_pattern = wrap_type<ov::op::v1::Softmax, ov::op::v8::Softmax>({matmul_pattern0});

    // Input Value
    auto input2 = any_input(has_static_rank());
    auto matmul_pattern1 = wrap_type<ov::op::v0::MatMul>({softmax_pattern, input2});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(matmul_pattern0));
        OPENVINO_ASSERT(pattern_map.count(softmax_pattern));
        OPENVINO_ASSERT(pattern_map.count(matmul_pattern1));

        auto matmul0 = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_pattern0).get_node_shared_ptr());
        if (!matmul0 || transformation_callback(matmul0))
            return false;

        auto input_q = pattern_map.at(input0).get_node_shared_ptr();
        auto input_k = matmul0->get_input_node_shared_ptr(1);
        auto input_v = pattern_map.at(input2).get_node_shared_ptr();

        // Check shape : currently MHA primtive supports limited shape and format
        auto shape_q = pattern_map.at(input0).get_partial_shape();
        auto shape_k = matmul0->get_input_partial_shape(1);
        auto shape_v = pattern_map.at(input2).get_partial_shape();
        auto shape_p = matmul0->get_output_partial_shape(0);
        auto target_input_shape = ov::PartialShape(MatchingInputSize);
        if (shape_q != target_input_shape || shape_k != target_input_shape || shape_v != target_input_shape ||
            shape_p != ov::PartialShape(MatchingPSize))
            return false;

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
            return transpose;
        };

        auto create_reshape = [this, &new_ops ](const ov::Output<ov::Node>& node,
                                                 const ov::Shape& new_shape,
                                                 const std::string& reshape_name) {
            auto reshape = std::make_shared<ov::op::v1::Reshape>(
                            node,
                            ov::op::v0::Constant::create(ov::element::i64,
                                                            ov::Shape{new_shape.size()},
                                                            new_shape),
                            false);
            reshape->set_friendly_name(reshape_name);
            return reshape;
        };

        // Transpose 'Key' input to put into MHA primitive which does not support transpose of input buffers
        auto transposed_input_k = create_transpose(input_k, matmul0->get_friendly_name() + "/transpose_k");
        transposed_input_k->set_output_type(0, matmul0->get_output_element_type(0), transposed_input_k->get_output_partial_shape(0));
        new_ops.push_back(transposed_input_k);
        auto transposed_shape_k = transposed_input_k->get_output_partial_shape(0);

        // Reshape to use current MHA primitive which only supports limited format and size
        auto new_input_q = create_reshape(input_q, ov::Shape(ReshapeMatchingInputSize), matmul0->get_friendly_name() + "/mha/reshape_q");
        new_ops.push_back(new_input_q);
        auto new_input_k = create_reshape(transposed_input_k, ov::Shape(ReshapeMatchingTransPosedInputSize),
                                            matmul0->get_friendly_name() + "/mha/reshape_k");
        new_ops.push_back(new_input_k);
        auto new_input_v = create_reshape(input_v, ov::Shape(ReshapeMatchingInputSize), matmul0->get_friendly_name() + "/mha/reshape_v");
        new_ops.push_back(new_input_v);

        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_pattern1).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul))
            return false;

        ov::NodeVector nodes_to_copy_info{pattern_map.at(matmul_pattern0).get_node_shared_ptr(),
                                          pattern_map.at(softmax_pattern).get_node_shared_ptr(),
                                          pattern_map.at(matmul_pattern1).get_node_shared_ptr()};

        std::shared_ptr<ov::Node> mha = std::make_shared<op::MhaFusion>(new_input_q,
                                                                        new_input_k,
                                                                        new_input_v,
                                                                        matmul->get_output_element_type(0));
        mha->set_friendly_name(matmul->get_friendly_name()+ "/mha");
        for (size_t i=0 ; i < mha->get_output_size() ; i++) {
            mha->set_output_type(i, matmul->get_output_element_type(i), ov::Shape(ReshapeMatchingOutputSize));
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
