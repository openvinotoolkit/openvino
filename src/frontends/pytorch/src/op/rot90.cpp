// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"

#include "openvino/op/reverse.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "utils.hpp"

namespace ov{
namespace frontend{
namespace pytorch{
namespace op{
using namespace ov::op;
OutputVector translate_rot90(const NodeContext& context){
    
    num_inputs_check(context,1,3);
    auto tensor = context.get_input(0);
    auto k = context.input_is_none(1) ? 1 : context.const_input<int64_t>(1);
    const auto dims = context.input_is_none(2) ? context.mark_node(v0::Constant::create(element::i64, {2}, 
            {0, 1})) : context.get_input(2);
    k = (k%4+4)%4;
    auto zero = v0::Constant::create(element::i64, Shape{}, {0});
    auto one = v0::Constant::create(element::i64, Shape{}, {1});
    // normalizing dims to handle negative values
    auto shape_of = context.mark_node(std::make_shared<v3::ShapeOf>(tensor, element::i64));
    auto rank_node = context.mark_node(std::make_shared<v3::ShapeOf>(shape_of, element::i64));
    auto rank_scalar = context.mark_node(std::make_shared<v0::Squeeze>(rank_node, 
        v0::Constant::create(element::i64, Shape{}, {0})
    ));
    // create a tensor of dims shape to hold rank
    auto rank_bcast = context.mark_node(std::make_shared<v3::Broadcast>(
        rank_scalar,
        std::make_shared<v3::ShapeOf>(dims,element::i64)
    ));
    // normalization of dims is done here
    auto dim_mod = context.mark_node(std::make_shared<v1::Mod>(dims,rank_bcast));
    auto dim_mod_plus_rank = context.mark_node(std::make_shared<v1::Add>(dim_mod,rank_bcast));
    auto normalized_dims = context.mark_node(std::make_shared<v1::Mod>(dim_mod_plus_rank,rank_bcast));
    // using these scalars for creating axes for flipping
    auto dim1_index = v0::Constant::create(element::i64,{1},{1}); 
    auto dim1_axis = context.mark_node(std::make_shared<v8::Gather>(
        normalized_dims,dim1_index,zero
    ));
    auto dim0_index = v0::Constant::create(element::i64,{1},{0}); 
    auto dim0_axis = context.mark_node(std::make_shared<v8::Gather>(
        normalized_dims,dim0_index,zero
    ));
    // reverse order matrix for transpose permutation
    auto reverse_order = context.mark_node(std::make_shared<v1::Reverse>(
        normalized_dims,
        v0::Constant::create(element::i64,{1},{0}),
        v1::Reverse::Mode::INDEX
    ));
    // cases
    if(k==1){
        auto flipped = context.mark_node(std::make_shared<v1::Reverse>(tensor, dim1_axis, v1::Reverse::Mode::INDEX));
        auto perm = context.mark_node(std::make_shared<v4::Range>(zero,rank_scalar,one,element::i64));
        auto dims_perm = context.mark_node(std::make_shared<v3::ScatterUpdate>(
            perm,
            normalized_dims,
            reverse_order,
            zero
        ));
        auto transposed = context.mark_node(
        std::make_shared<v1::Transpose>(flipped, dims_perm));
        return {transposed};
    }
    if(k==2){
        auto flipped = context.mark_node(
            std::make_shared<v1::Reverse>(tensor, normalized_dims, v1::Reverse::Mode::INDEX));
        return {flipped};
    }
    if(k==3){
        auto flipped = context.mark_node(std::make_shared<v1::Reverse>(tensor, dim0_axis, v1::Reverse::Mode::INDEX));
        auto perm = context.mark_node(std::make_shared<v4::Range>(zero,rank_scalar,one,element::i64));
        auto dims_perm = context.mark_node(std::make_shared<v3::ScatterUpdate>(
            perm,
            normalized_dims,
            reverse_order,
            zero
        ));
        auto transposed = context.mark_node(
        std::make_shared<v1::Transpose>(flipped, dims_perm));
        return {transposed};
    }
    return {tensor};
};
} // namespace op;
} // namespace pytorch
} // namespace frontend
} // namespace ov