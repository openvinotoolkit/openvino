// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> prepare_source(const NodeContext& context,
                            const Output<Node>& src,
                            const Output<Node>& index,
                            const Output<Node>& input) {
    const auto& src_partial_shape = src.get_partial_shape();
    auto index_shape_rank = get_shape_rank(context, index);
    auto index_shape = std::get<0>(index_shape_rank);
    auto index_rank = std::get<1>(index_shape_rank);

    // Source input can be either Tensor which should be passed in original shape or Scalar that should be broadcasted
    // into shape of indices.
    // TODO: Figure out way to dynamically broadcast scalar src only, without affecting Tensor src. Current
    // implementation will fail if Scalar source would have dynamic rank.
    auto _src = std::move(src);
    if (src_partial_shape.rank().is_static() && src_partial_shape.rank().get_length() == 0) {
        _src = context.mark_node(std::make_shared<v3::Broadcast>(_src, index_shape));
    }

    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto zeros = context.mark_node(std::make_shared<v3::Broadcast>(const_0, index_rank));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto ones = context.mark_node(std::make_shared<v3::Broadcast>(const_1, index_rank));
    // In torch indices can be of different shape than source tensor. Create slice to trim source tensor to shape of
    // indices.
    auto src_pruned = context.mark_node(std::make_shared<v8::Slice>(_src, zeros, index_shape, ones));

    auto src_input_dtype = context.mark_node(std::make_shared<v1::ConvertLike>(src_pruned, input));
    return src_input_dtype;
};

const v12::ScatterElementsUpdate::Reduction get_reduction_mode(const std::string& pt_reduce_mode) {
    static const std::unordered_map<std::string, v12::ScatterElementsUpdate::Reduction> TORCH_REDUCTION_TO_OV{
        {"add", v12::ScatterElementsUpdate::Reduction::SUM},
        {"multiply", v12::ScatterElementsUpdate::Reduction::PROD},
        {"sum", v12::ScatterElementsUpdate::Reduction::SUM},
        {"prod", v12::ScatterElementsUpdate::Reduction::PROD},
        {"mean", v12::ScatterElementsUpdate::Reduction::MEAN},
        {"amax", v12::ScatterElementsUpdate::Reduction::MAX},
        {"amin", v12::ScatterElementsUpdate::Reduction::MIN}};

    PYTORCH_OP_CONVERSION_CHECK(TORCH_REDUCTION_TO_OV.count(pt_reduce_mode),
                                "Unknown reduction mode: ",
                                pt_reduce_mode);
    auto reduction = TORCH_REDUCTION_TO_OV.at(pt_reduce_mode);
    return reduction;
}
};  // namespace

OutputVector translate_scatter(const NodeContext& context) {
    // Out-of-place schema
    // aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor:
    // aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor:
    // aten::scatter.src_out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
    // aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor:
    // aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor:
    // aten::scatter.value_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
    // aten::scatter.value_reduce_out(Tensor self, int dim, Tensor index, Scalar value, *, str reduce, Tensor(a!) out)
    // -> Tensor(a!)

    // Inplace schema
    // aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!):
    // aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!):
    // aten::scatter_.reduce(Tensor(a!) self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor(a!):
    // aten::scatter_.value_reduce(Tensor(a!) self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor(a!):
    num_inputs_check(context, 4, 6);
    auto input = context.get_input(0);
    auto dim = context.get_input(1);
    auto index = context.mark_node(std::make_shared<v0::Convert>(context.get_input(2), element::i32));
    auto src = context.get_input(3);

    auto reduction = v12::ScatterElementsUpdate::Reduction::NONE;
    auto input_num = context.get_input_size();
    // 5 argument can be reduction represented as string or out represented as Tensor
    if (input_num > 4 && !context.input_is_none(4) && context.get_input_type(4).is<type::Str>()) {
        auto reduce_mode = context.const_input<std::string>(4);
        reduction = get_reduction_mode(reduce_mode);
    }

    auto src_input_dtype = prepare_source(context, src, index, input);
    auto res =
        context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(input, index, src_input_dtype, dim, reduction));
    if (input_num == 6 || (input_num == 5 && !context.get_input_type(4).is<type::Str>())) {
        context.mutate_input(input_num - 1, res);
    }
    return {res};
};

OutputVector translate_scatter_reduce(const NodeContext& context) {
    // Out-of-place schema
    // aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True)
    // -> Tensor: aten::scatter_reduce.two_out(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool
    // include_self=True, Tensor(a!) out) -> Tensor(a!)

    // Inplace schema
    // aten::scatter_reduce_.two(Tensor(a!) self, int dim, Tensor index, Tensor src, str reduce, *, bool
    // include_self=True) -> Tensor(a!)
    num_inputs_check(context, 6, 7);
    auto input = context.get_input(0);
    auto dim = context.get_input(1);
    auto index = context.mark_node(std::make_shared<v0::Convert>(context.get_input(2), element::i32));
    auto src = context.get_input(3);
    auto reduce_mode = context.const_input<std::string>(4);
    auto reduction = get_reduction_mode(reduce_mode);
    auto include_self = context.const_input<bool>(5);
    auto src_input_dtype = prepare_source(context, src, index, input);
    auto scatter_result = context.mark_node(
        std::make_shared<v12::ScatterElementsUpdate>(input, index, src_input_dtype, dim, reduction, include_self));
    if (!context.input_is_none(6)) {
        context.mutate_input(6, scatter_result);
    }
    return {scatter_result};
};

OutputVector translate_scatter_add(const NodeContext& context) {
    // aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
    num_inputs_check(context, 4, 4);
    auto input = context.get_input(0);
    auto dim = context.get_input(1);
    auto index = context.mark_node(std::make_shared<v0::Convert>(context.get_input(2), element::i32));
    auto src = context.get_input(3);
    auto src_input_dtype = prepare_source(context, src, index, input);
    auto scatter_result =
        context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(input,
                                                                       index,
                                                                       src_input_dtype,
                                                                       dim,
                                                                       v12::ScatterElementsUpdate::Reduction::SUM));
    return {scatter_result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
