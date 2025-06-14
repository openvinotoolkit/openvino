// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/range.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_segment_mean_csr(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    
    auto src = context.get_input(0);
    auto indptr = context.get_input(1);
    
    auto indptr_shape = indptr.get_shape();
    auto indptr_size = ov::shape_size(indptr_shape);
    auto reshape_pattern = context.mark_node(std::make_shared<ov::op::v0::Constant>(
        element::i64, Shape{1}, std::vector<int64_t>{static_cast<int64_t>(indptr_size)}));
    auto indptr_flattened = context.mark_node(std::make_shared<ov::op::v1::Reshape>(indptr, reshape_pattern, false))->output(0);
    
    auto num_segments = indptr_size - 1;
    
    auto src_shape = src.get_shape();
    auto batch_size = src_shape[0];
    auto src_len = src_shape[1];
    auto feature_size = src_shape[2];
    
    auto result_shape = src_shape;
    result_shape[1] = num_segments;
    
    
    auto zeros = context.mark_node(std::make_shared<ov::op::v0::Constant>(
        src.get_element_type(), result_shape, std::vector<float>(ov::shape_size(result_shape), 0.0f)))->output(0);
    
    auto counts_shape = result_shape;
    auto counts = context.mark_node(std::make_shared<ov::op::v0::Constant>(
        src.get_element_type(), counts_shape, std::vector<float>(ov::shape_size(counts_shape), 0.0f)))->output(0);
    
    
    auto ones = context.mark_node(std::make_shared<ov::op::v0::Constant>(
        src.get_element_type(), src_shape, std::vector<float>(ov::shape_size(src_shape), 1.0f)))->output(0);
    
    auto start = context.mark_node(std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0))->output(0);
    auto stop = context.mark_node(std::make_shared<ov::op::v0::Constant>(
        element::i32, Shape{}, static_cast<int32_t>(src_len)))->output(0);
    auto step = context.mark_node(std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1))->output(0);
    auto indices = context.mark_node(std::make_shared<ov::op::v4::Range>(start, stop, step))->output(0);
    
    
    Output<Node> sums = zeros;
    Output<Node> counts_sum = counts;
    
    
    for (size_t seg_idx = 0; seg_idx < num_segments; seg_idx++) {
        
        auto start_idx = context.mark_node(std::make_shared<ov::op::v8::Gather>(
            indptr_flattened,
            context.mark_node(std::make_shared<ov::op::v0::Constant>(
                element::i64, Shape{}, static_cast<int64_t>(seg_idx)))->output(0),
            context.mark_node(std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0))->output(0),
            0))->output(0);
        
        auto end_idx = context.mark_node(std::make_shared<ov::op::v8::Gather>(
            indptr_flattened,
            context.mark_node(std::make_shared<ov::op::v0::Constant>(
                element::i64, Shape{}, static_cast<int64_t>(seg_idx + 1)))->output(0),
            context.mark_node(std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0))->output(0),
            0))->output(0);
        
       
        auto ge_mask = context.mark_node(std::make_shared<ov::op::v1::GreaterEqual>(
            indices, 
            context.mark_node(std::make_shared<ov::op::v1::Convert>(start_idx, element::i32))->output(0)))->output(0);
        
        auto lt_mask = context.mark_node(std::make_shared<ov::op::v1::Less>(
            indices, 
            context.mark_node(std::make_shared<ov::op::v1::Convert>(end_idx, element::i32))->output(0)))->output(0);
        
        auto mask = context.mark_node(std::make_shared<ov::op::v1::LogicalAnd>(ge_mask, lt_mask))->output(0);
        auto mask_f = context.mark_node(std::make_shared<ov::op::v1::Convert>(mask, src.get_element_type()))->output(0);
        
        auto mask_shape = std::vector<size_t>{1, src_len, 1};
        auto mask_reshape = context.mark_node(std::make_shared<ov::op::v1::Reshape>(
            mask_f,
            context.mark_node(std::make_shared<ov::op::v0::Constant>(
                element::i64, Shape{mask_shape.size()}, mask_shape))->output(0),
            false))->output(0);
        
        auto masked_src = context.mark_node(std::make_shared<ov::op::v1::Multiply>(src, mask_reshape))->output(0);
        auto masked_ones = context.mark_node(std::make_shared<ov::op::v1::Multiply>(ones, mask_reshape))->output(0);
        
       
        auto seg_idx_const = context.mark_node(std::make_shared<ov::op::v0::Constant>(
            element::i64, Shape{1}, std::vector<int64_t>{static_cast<int64_t>(seg_idx)}))->output(0);
        
        
        auto seg_counts = context.mark_node(std::make_shared<ov::op::v4::ReduceSum>(
            masked_ones, 
            context.mark_node(std::make_shared<ov::op::v0::Constant>(
                element::i32, Shape{1}, std::vector<int32_t>{1}))->output(0),
            false))->output(0);
        
        auto seg_sums = context.mark_node(std::make_shared<ov::op::v4::ReduceSum>(
            masked_src, 
            context.mark_node(std::make_shared<ov::op::v0::Constant>(
                element::i32, Shape{1}, std::vector<int32_t>{1}))->output(0),
            false))->output(0);
        
        
        
        auto indices_out = context.mark_node(std::make_shared<ov::op::v0::Constant>(
            element::i32, Shape{3}, std::vector<int32_t>{0, static_cast<int32_t>(seg_idx), 0}))->output(0);
        
        sums = context.mark_node(std::make_shared<ov::op::v7::ScatterElementsUpdate>(
            sums,
            indices_out,
            seg_sums,
            context.mark_node(std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1))->output(0)))->output(0);
        
        counts_sum = context.mark_node(std::make_shared<ov::op::v7::ScatterElementsUpdate>(
            counts_sum,
            indices_out,
            seg_counts,
            context.mark_node(std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1))->output(0)))->output(0);
    }
    
    auto epsilon = context.mark_node(std::make_shared<ov::op::v0::Constant>(
        src.get_element_type(), Shape{}, std::vector<float>{1e-10f}))->output(0);
    auto safe_counts = context.mark_node(std::make_shared<ov::op::v1::Add>(counts_sum, epsilon))->output(0);
    
    auto mean = context.mark_node(std::make_shared<ov::op::v1::Divide>(sums, safe_counts))->output(0);
    
    if (!context.input_is_none(2)) {
        context.mutate_input(2, mean);
    }
    
    return {mean};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov