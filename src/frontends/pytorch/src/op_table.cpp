// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op_table.hpp"

#include <openvino/opsets/opset8.hpp>

#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

#define OP_CONVERTER(op) OutputVector op(NodeContext& node)

OP_CONVERTER(translate_loop);
OP_CONVERTER(translate_if);

OutputVector relu(NodeContext& context) {
    return {context.mark_node(std::make_shared<opset8::Relu>(context.get_input(0)))};
};
OutputVector add(NodeContext& context) {
    return {context.mark_node(std::make_shared<opset8::Add>(context.get_input(0), context.get_input(1)))};
};

OutputVector mul(NodeContext& context) {
    return {context.mark_node(std::make_shared<opset8::Multiply>(context.get_input(0), context.get_input(1)))};
}

OutputVector hardtanh(NodeContext& context) {
    float min = -1;
    float max = 1;
    if (!context.input_is_none(1)) {
        min = context.const_input<float>(1);
    }
    if (!context.input_is_none(2)) {
        max = context.const_input<float>(2);
    }
    return {context.mark_node(std::make_shared<opset8::Clamp>(context.get_input(0), min, max))};
}

OutputVector hardswish(NodeContext& context) {
    return {context.mark_node(std::make_shared<opset8::HSwish>(context.get_input(0)))};
}

const std::map<int, element::Type> type_map{
    {0, element::u8},
    {6, element::f32},
};

OutputVector translate_aten_to(NodeContext& context) {
    int dtype_idx;
    int non_blocking_idx;
    int copy_idx;
    int memory_format_idx;
    if (context.get_input_size() == 5) {
        // aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None)
        // -> (Tensor(a))
        dtype_idx = 1;
        non_blocking_idx = 2;
        copy_idx = 3;
        memory_format_idx = 4;
    } else if (context.get_input_size() == 6) {
        // aten::to.device(Tensor(a) self, Device device, int dtype, bool non_blocking=False, bool copy=False, int?
        // memory_format=None) -> (Tensor(a)).
        // Skipping "device" input.
        dtype_idx = 2;
        non_blocking_idx = 3;
        copy_idx = 4;
        memory_format_idx = 5;
    } else {
        FRONT_END_OP_CONVERSION_CHECK(false, "Unknown aten::to format");
    }
    // TODO: do we need to check these inputs?
    // OV_FRONTEND_REQUIRE(context.const_input<bool>(non_blocking_idx) == false);
    // OV_FRONTEND_REQUIRE(context.const_input<bool>(copy_idx) == false);
    // OV_FRONTEND_REQUIRE(context.input_is_none(memory_format_idx));
    auto dtype_ext_node = context.get_input_from_visible_context(dtype_idx).get_node_shared_ptr();
    auto dtype_tensor = context.get_input(dtype_idx);
    auto dtype_fw_node = std::dynamic_pointer_cast<PtFrameworkNode>(dtype_tensor.get_node_shared_ptr());
    Output<Node> cast;
    if (dtype_fw_node && dtype_fw_node->get_op_type() == "prim::dtype") {
        auto type_input = dtype_fw_node->input(0).get_source_output();
        cast = context.mark_node(std::make_shared<opset8::ConvertLike>(context.get_input(0), type_input));
    } else if (std::dynamic_pointer_cast<opset8::Constant>(dtype_ext_node)) {
        auto pt_type = context.const_input<int64_t>(dtype_idx);
        FRONT_END_OP_CONVERSION_CHECK(type_map.count(pt_type), "Unknown type in aten::to: ", pt_type);
        auto dtype = type_map.at(pt_type);
        cast = context.mark_node(std::make_shared<opset8::Convert>(context.get_input(0), dtype));
    } else {
        cast = context.mark_node(std::make_shared<opset8::ConvertLike>(context.get_input(0), dtype_tensor));
    }
    return {cast};
}

OutputVector translate_as_tensor(NodeContext& context) {
    auto dtype_ext_node = context.get_input_from_visible_context(1).get_node_shared_ptr();
    auto dtype_tensor = context.get_input(1);
    auto dtype_fw_node = std::dynamic_pointer_cast<PtFrameworkNode>(dtype_tensor.get_node_shared_ptr());
    Output<Node> cast;
    if (dtype_fw_node && dtype_fw_node->get_op_type() == "prim::dtype") {
        auto type_input = dtype_fw_node->input(0).get_source_output();
        cast = context.mark_node(std::make_shared<opset8::ConvertLike>(context.get_input(0), type_input));
    } else if (std::dynamic_pointer_cast<opset8::Constant>(dtype_ext_node)) {
        auto pt_type = context.const_input<int64_t>(1);
        FRONT_END_OP_CONVERSION_CHECK(type_map.count(pt_type), "Unknown type in aten::as_tensor: ", pt_type);
        auto dtype = type_map.at(pt_type);
        cast = context.mark_node(std::make_shared<opset8::Convert>(context.get_input(0), dtype));
    }
    // OV_FRONTEND_REQUIRE(context.input_is_none(2));  // device: no need to check
    // auto new_shape_const = context.mark_node(opset8::Constant::create(element::i64, {1}, {1}));
    // return { context.mark_node(std::make_shared<opset8::Reshape>(cast,
    // new_shape_const->output(0), true)) };
    return {cast};
}

}  // namespace op

const std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        {"aten::relu", op::relu},
        {"aten::relu_", inplace_op<op::relu>},

        {"aten::conv2d",
         [](NodeContext& context) -> OutputVector {
             auto strides = context.const_input<Strides>(3);
             auto pads_begin = context.const_input<CoordinateDiff>(4);  // FIXME: The same 4 is used twice
             auto pads_end = context.const_input<CoordinateDiff>(4);    // FIXME: The same 4 is used twice
             auto dilations = context.const_input<Strides>(5);
             auto groups = context.const_input<int64_t>(6);

             std::shared_ptr<ov::Node> conv;
             if (groups == 1) {
                 conv = std::make_shared<opset8::Convolution>(context.get_input(0),
                                                              context.get_input(1),
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations);
             } else {
                 conv = std::make_shared<opset8::GroupConvolution>(
                     context.get_input(0),
                     reshape_kernel_for_group(context, context.get_input(0), context.get_input(1), groups),
                     strides,
                     pads_begin,
                     pads_end,
                     dilations);
             }

             // FIXME: Doesn't work for dynamic rank
             // FIXME: Works for 2D convolutions only
             return {context.mark_output(make_optional_bias(conv, context, 2, {-2, -1}))};
         }},

        {"aten::_convolution",
         [](NodeContext& context) -> OutputVector {
             bool transposed = context.const_input<bool>(6);
             // TODO: Handle this temporary limitation
             OV_FRONTEND_REQUIRE(!transposed);

             auto strides = context.const_input<Strides>(3);
             auto pads_begin = context.const_input<CoordinateDiff>(4);  // FIXME: The same 4 is used twice
             auto pads_end = context.const_input<CoordinateDiff>(4);    // FIXME: The same 4 is used twice
             auto dilations = context.const_input<Strides>(5);
             // TODO: Handle skipped input 7 (6 was used above) -- what is it for?
             auto groups = context.const_input<int64_t>(8);

             std::shared_ptr<ov::Node> conv;
             if (groups == 1) {
                 conv = std::make_shared<opset8::Convolution>(context.get_input(0),
                                                              context.get_input(1),
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations);
             } else {
                 conv = std::make_shared<opset8::GroupConvolution>(
                     context.get_input(0),
                     context.mark_output(
                         reshape_kernel_for_group(context, context.get_input(0), context.get_input(1), groups)),
                     strides,
                     pads_begin,
                     pads_end,
                     dilations);
             }

             // FIXME: Doesn't work for dynamic rank
             // FIXME: Works for 2D convolutions only
             return {context.mark_output(make_optional_bias(conv, context, 2, {-2, -1}))};
         }},

        {"aten::batch_norm",
         [](NodeContext& context) -> OutputVector {
             auto training = context.const_input<bool>(5);
             OV_FRONTEND_REQUIRE(!training);  // TODO: support bn training
             return {context.mark_node(std::make_shared<opset8::BatchNormInference>(
                 context.get_input(0),
                 context.get_input(1),
                 context.get_input(2),
                 context.get_input(3),
                 context.get_input(4),
                 context.const_input<float>(7)  // epsilon
                 ))};
         }},

        {"aten::layer_norm",
         [](NodeContext& context) -> OutputVector {
             auto normalized_shape = context.const_input<Shape>(1);
             auto in_pshape_last_dim = *context.get_input(0).get_partial_shape().rbegin();
             OV_FRONTEND_REQUIRE(normalized_shape.size() == 1 && in_pshape_last_dim.is_static() &&
                                 static_cast<uint64_t>(in_pshape_last_dim.get_length()) == normalized_shape.back());
             auto eps = context.const_input<float>(4);
             auto axes = context.mark_node(
                 opset8::Constant::create(element::i64, Shape{1}, {-1}));  // TODO: support any dimention
             auto mvn = context.mark_node(
                 std::make_shared<opset8::MVN>(context.get_input(0), axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT));
             std::shared_ptr<ov::Node> out_node = std::dynamic_pointer_cast<ov::Node>(mvn);
             if (!context.input_is_none(2)) {
                 auto mul = std::make_shared<opset8::Multiply>(out_node, context.get_input(2));
                 out_node = std::dynamic_pointer_cast<ov::Node>(mul);
             }
             if (!context.input_is_none(3)) {
                 auto add = std::make_shared<opset8::Add>(out_node, context.get_input(3));
                 out_node = std::dynamic_pointer_cast<ov::Node>(add);
             }
             return {context.mark_node(out_node)};
         }},

        {"aten::add", op::add},
        {"aten::add_", inplace_op<op::add>},

        {"aten::mul", op::mul},
        {"aten::mul_", inplace_op<op::mul>},

        {"aten::div",
         [](NodeContext& context) -> OutputVector {
             auto pythondiv = false;
             if (!context.input_is_none(2)) {
                 auto rounding_mode = context.const_input<std::string>(2);
                 if (rounding_mode == "floor") {
                     pythondiv = true;
                 } else if (rounding_mode == "trunc") {
                     pythondiv = true;
                     // break;
                 }
             }
             return {context.mark_node(
                 std::make_shared<opset8::Divide>(context.get_input(0), context.get_input(1), pythondiv))};
         }},

        {"aten::tanh",
         [](NodeContext& context) -> OutputVector {
             return {context.mark_node(std::make_shared<opset8::Tanh>(context.get_input(0)))};
         }},

        {"aten::elu",
         [](NodeContext& context) -> OutputVector {
             auto alpha = context.const_input<float>(1);
             return {context.mark_node(std::make_shared<opset8::Elu>(context.get_input(0), alpha))};
         }},

        {"aten::sigmoid",
         [](NodeContext& context) -> OutputVector {
             return {context.mark_node(std::make_shared<opset8::Sigmoid>(context.get_input(0)))};
         }},

        {"aten::gelu",
         [](NodeContext& context) -> OutputVector {
             return {context.mark_node(std::make_shared<opset8::Gelu>(context.get_input(0)))};
         }},

        {"aten::sqrt",
         [](NodeContext& context) -> OutputVector {
             return {context.mark_node(std::make_shared<opset8::Sqrt>(context.get_input(0)))};
         }},

        {"aten::abs",
         [](NodeContext& context) -> OutputVector {
             return {context.mark_node(std::make_shared<opset8::Abs>(context.get_input(0)))};
         }},

        {"aten::square",
         [](NodeContext& context) -> OutputVector {
             auto input_0 = context.get_input(0);
             auto const_2 = context.mark_node(opset8::Constant::create(input_0.get_element_type(), Shape{1}, {2}));
             return {context.mark_node(std::make_shared<opset8::Power>(input_0, const_2))};
         }},

        {"aten::hardtanh", op::hardtanh},
        {"aten::hardtanh_", inplace_op<op::hardtanh>},

        {"aten::hardsigmoid",
         [](NodeContext& context) -> OutputVector {
             return {context.mark_node(std::make_shared<opset8::HSigmoid>(context.get_input(0)))};
         }},

        {"aten::hardswish", op::hardswish},
        {"aten::hardswish_", inplace_op<op::hardswish>},

        {"aten::silu_",
         [](NodeContext& context) -> OutputVector {
             auto swish = std::make_shared<opset8::Swish>(context.get_input(0));
             context.mutate_input(0, swish);
             return {context.mark_node(swish)};
         }},

        {"aten::relu6",
         [](NodeContext& context) -> OutputVector {
             return {context.mark_node(std::make_shared<opset8::Clamp>(context.get_input(0), 0., 6.))};
         }},

        {"aten::softmax",
         [](NodeContext& context) -> OutputVector {
             auto axis = context.const_input<int64_t>(1);
             if (axis < 0) {
                 auto in_rank = context.get_input(0).get_partial_shape().rank();
                 OV_FRONTEND_REQUIRE(in_rank.is_static());
                 axis = in_rank.get_length() + axis;
             }
             return {
                 context.mark_node(std::make_shared<opset8::Softmax>(context.get_input(0), static_cast<size_t>(axis)))};
         }},

        {"aten::cat",
         [](NodeContext& context) -> OutputVector {
             // aten::cat needs a special handling since it takes a Tensor[] as
             // input. We set the inputs of ListConstruct as the inputs of cat.
             //
             // Pytorch IR:                              LLGA sees:
             //     %a    %b     %c          %dim              %a    %b    %c
             //      \     |     /             |                \     |    /
             //   prim::ListConstruct   prim::Constant     llga::Concat[axis=%dim]
             //                    \      /
             //                    aten::cat
             auto listConstruct = context.get_input(0).get_node();
             auto listConstruct_fw_node = dynamic_cast<PtFrameworkNode*>(listConstruct);
             OV_FRONTEND_REQUIRE(listConstruct_fw_node);
             OV_FRONTEND_REQUIRE(listConstruct_fw_node->get_op_type() == "prim::ListConstruct");
             auto axis = context.const_input<int64_t>(1);
             OutputVector inputs;
             for (auto& input : listConstruct->inputs()) {
                 inputs.push_back(input.get_source_output());
             }
             auto result = context.mark_node(std::make_shared<opset8::Concat>(inputs, axis));
             // TODO: do we really need to do that?
             // auto list_set = listConstruct_fw_node->get_rt_info()["pt_node"].as<std::set<const Node*>>();
             // result->get_rt_info()["pt_node"].as<std::set<const Node*>>().insert(list_set.begin(), list_set.end());
             return {result};
         }},

        {"aten::matmul",
         [](NodeContext& context) -> OutputVector {
             return {context.mark_node(std::make_shared<opset8::MatMul>(context.get_input(0), context.get_input(1)))};
         }},

        {"aten::mm",
         [](NodeContext& context) -> OutputVector {
             return {context.mark_node(std::make_shared<opset8::MatMul>(context.get_input(0), context.get_input(1)))};
         }},

        {"aten::linear",
         [](NodeContext& context) -> OutputVector {
             auto matmul = std::make_shared<opset8::MatMul>(context.get_input(0), context.get_input(1), false, true);
             return {context.mark_output(make_optional_bias(matmul, context, 2))};
         }},

        {"aten::max_pool2d",
         [](NodeContext& context) -> OutputVector {
             auto kernel = context.const_input<Shape>(1);
             auto strides = context.const_input<Strides>(2);
             auto pads_begin = context.const_input<Shape>(3);  // FIXME: The same 3 is used twice
             auto pads_end = context.const_input<Shape>(3);    // FIXME: The same 3 is used twice
             auto dilations = context.const_input<Strides>(4);
             auto rounding_type =
                 context.const_input<bool>(5) ? ov::op::RoundingType::CEIL : ov::op::RoundingType::FLOOR;

             // TODO: Upgrade to opset8::MaxPool to use dilations; for now we suppose they are all zeros
             return {context.mark_node(std::make_shared<opset8::MaxPool>(context.get_input(0),
                                                                         strides,
                                                                         dilations,
                                                                         pads_begin,
                                                                         pads_end,
                                                                         kernel,
                                                                         rounding_type))};
         }},

        {"aten::avg_pool2d",
         [](NodeContext& context) -> OutputVector {
             auto kernel = context.const_input<Shape>(1);
             auto strides = context.const_input<Strides>(2);
             auto pads_begin = context.const_input<Shape>(3);  // FIXME: The same 3 is used twice
             auto pads_end = context.const_input<Shape>(3);    // FIXME: The same 3 is used twice
             auto rounding_type =
                 context.const_input<bool>(4) ? ov::op::RoundingType::CEIL : ov::op::RoundingType::FLOOR;
             auto exclude_pad = !context.const_input<bool>(5);
             // TODO: support divisor override
             // auto divisor_override = context.const_input<int64_t>(6);

             return {context.mark_node(std::make_shared<opset8::AvgPool>(context.get_input(0),
                                                                         strides,
                                                                         pads_begin,
                                                                         pads_end,
                                                                         kernel,
                                                                         exclude_pad,
                                                                         rounding_type))};
         }},

        {"aten::adaptive_avg_pool2d",
         [](NodeContext& context) -> OutputVector {
             return {context.mark_node(
                 std::make_shared<opset8::AdaptiveAvgPool>(context.get_input(0), context.get_input(1)))};
         }},

        {"aten::adaptive_max_pool2d",
         [](NodeContext& context) -> OutputVector {
             auto adaptive_max_pool = context.mark_node(
                 std::make_shared<opset8::AdaptiveMaxPool>(context.get_input(0), context.get_input(1)));
             auto return_indices = context.const_input<bool>(2);
             OutputVector res{adaptive_max_pool->output(0)};
             if (return_indices) {
                 res.push_back(adaptive_max_pool->output(1));
             }
             return res;
         }},

        {"aten::mean",
         [](NodeContext& context) -> OutputVector {
             auto keep_dims = context.const_input<bool>(2);
             OV_FRONTEND_REQUIRE(context.input_is_none(3));
             return {context.mark_node(
                 std::make_shared<opset8::ReduceMean>(context.get_input(0), context.get_input(1), keep_dims))};
         }},

        {"aten::flatten",
         [](NodeContext& context) -> OutputVector {
             auto start_dim = context.const_input<int64_t>(1);
             auto end_dim = context.const_input<int64_t>(2);
             auto data_pshape = context.get_input(0).get_partial_shape();
             OV_FRONTEND_REQUIRE(data_pshape.rank().is_static());  // TODO: support dynamic rank
             auto rank = data_pshape.rank().get_length();
             if (start_dim < 0) {
                 start_dim = rank + start_dim;
             }
             if (end_dim < 0) {
                 end_dim = rank + end_dim;
             }
             OV_FRONTEND_REQUIRE(start_dim < end_dim);
             auto delta = end_dim - start_dim;
             std::vector<int64_t> new_shape(rank - delta, 0);
             new_shape[start_dim] = -1;
             auto new_shape_const =
                 context.mark_node(opset8::Constant::create(element::i64, {new_shape.size()}, new_shape));
             return {context.mark_node(std::make_shared<opset8::Reshape>(context.get_input(0), new_shape_const, true))};
         }},

        {"prim::NumToTensor",
         [](NodeContext& context) -> OutputVector {
             // Do nothing // TODO: Really? Should we produce scalar tensor with shape [] instead of custom PT type?
             return {context.mark_node(context.get_input(0).get_node_shared_ptr())};
         }},

        {"aten::contiguous",
         [](NodeContext& context) -> OutputVector {
             // Do nothing
             return {context.mark_node(context.get_input(0).get_node_shared_ptr())};
         }},

        {"aten::as_tensor", op::translate_as_tensor},

        {"aten::Int",
         [](NodeContext& context) -> OutputVector {
             return {context.mark_node(std::make_shared<opset8::Convert>(context.get_input(0), element::i64))};
         }},

        {"aten::to", op::translate_aten_to},

        {"aten::permute",
         [](NodeContext& context) -> OutputVector {
             return {
                 context.mark_node(std::make_shared<opset8::Transpose>(context.get_input(0), context.get_input(1)))};
         }},

        {"aten::embedding",
         [](NodeContext& context) -> OutputVector {
             // TODO: find out the meaning of input idx 2
             OV_FRONTEND_REQUIRE(context.const_input<bool>(3) == false);
             OV_FRONTEND_REQUIRE(context.const_input<bool>(4) == false);
             auto axis_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
             return {context.mark_node(
                 std::make_shared<opset8::Gather>(context.get_input(0), context.get_input(1), axis_0))};
         }},

        {"aten::transpose",
         [](NodeContext& context) -> OutputVector {
             auto dim0 = context.const_input<int64_t>(1);
             auto dim1 = context.const_input<int64_t>(2);
             auto data_pshape = context.get_input(0).get_partial_shape();
             auto rank = data_pshape.rank();
             OV_FRONTEND_REQUIRE(rank.is_static());
             auto _rank = rank.get_length();
             if (dim0 < 0) {
                 dim0 = _rank + dim0;
             }
             if (dim1 < 0) {
                 dim1 = _rank + dim1;
             }
             OV_FRONTEND_REQUIRE(dim0 > 0 && dim1 > 0);
             OV_FRONTEND_REQUIRE(dim0 < _rank && dim1 < _rank);
             std::vector<int64_t> order(_rank, 0);
             std::iota(order.begin(), order.end(), 0);
             std::swap(order[dim0], order[dim1]);
             auto order_const = context.mark_node(opset8::Constant::create(element::i64, {order.size()}, order));
             return {context.mark_node(std::make_shared<opset8::Transpose>(context.get_input(0), order_const))};
         }},

        {"aten::size",
         [](NodeContext& context) -> OutputVector {
             auto shape = context.mark_node(std::make_shared<opset8::ShapeOf>(context.get_input(0)));
             if (context.input_is_none(1)) {
                 return shape->outputs();
             } else {
                 auto axis_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
                 return {context.mark_node(std::make_shared<opset8::Gather>(shape, context.get_input(1), axis_0))};
             }
         }},

        {"aten::view",
         [](NodeContext& context) -> OutputVector {
             auto shape_node = context.get_input(1).get_node();
             auto shape_node_fw_node = dynamic_cast<PtFrameworkNode*>(shape_node);
             std::shared_ptr<ov::Node> reshape;
             if (shape_node_fw_node && shape_node_fw_node->get_decoder()->get_op_type() == "prim::ListConstruct") {
                 // TODO: maybe use pt shape instead of whole shape subgraph, because it may be more efficent
                 OutputVector inputs;
                 auto axis_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
                 for (auto& input : shape_node->inputs()) {
                     auto rank = input.get_partial_shape().rank();
                     OV_FRONTEND_REQUIRE(rank.is_dynamic() || rank.get_length() == 0);
                     auto unsqueeze =
                         context.mark_node(std::make_shared<opset8::Unsqueeze>(input.get_source_output(), axis_0));
                     inputs.push_back(unsqueeze);
                 }
                 auto concat = context.mark_node(std::make_shared<opset8::Concat>(inputs, 0));
                 reshape = context.mark_node(std::make_shared<opset8::Reshape>(context.get_input(0), concat, false));
                 auto list_set = shape_node_fw_node->get_rt_info()["pt_node"].as<std::set<const Node*>>();
                 reshape->get_rt_info()["pt_node"].as<std::set<const Node*>>().insert(list_set.begin(), list_set.end());
             } else {
                 reshape = context.mark_node(
                     std::make_shared<opset8::Reshape>(context.get_input(0), context.get_input(1), false));
             }
             return {reshape};
         }},

        {"aten::unsqueeze",
         [](NodeContext& context) -> OutputVector {
             return {
                 context.mark_node(std::make_shared<opset8::Unsqueeze>(context.get_input(0), context.get_input(1)))};
         }},

        {"aten::rsub",
         [](NodeContext& context) -> OutputVector {
             // reverse aten::sub other - self * alpha
             auto alpha_casted = context.mark_node(
                 std::make_shared<opset8::Convert>(context.get_input(2), context.get_input(0).get_element_type()));
             auto alpha_mul = context.mark_node(std::make_shared<opset8::Multiply>(context.get_input(0), alpha_casted));
             return {context.mark_node(std::make_shared<opset8::Subtract>(context.get_input(1), alpha_mul))};
         }},

        {"aten::slice",
         [](NodeContext& context) -> OutputVector {
             // aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> (t[])
             // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> (Tensor(a))
             ov::Output<ov::Node> dim;
             ov::Output<ov::Node> start;
             ov::Output<ov::Node> end;
             ov::Output<ov::Node> step;
             auto axis_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
             if (context.get_input_size() == 5) {
                 dim = context.get_input(1);
                 start = context.get_input(2);
                 end = context.get_input(3);
                 step = context.get_input(4);
                 if (dim.get_partial_shape().rank().is_dynamic() || dim.get_partial_shape().rank().get_length() == 0) {
                     dim = context.mark_node(std::make_shared<opset8::Unsqueeze>(dim, axis_0));
                 }
             } else if (context.get_input_size() == 4) {
                 start = context.get_input(1);
                 end = context.get_input(2);
                 step = context.get_input(3);
                 dim = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {0}));
             } else {
                 FRONT_END_OP_CONVERSION_CHECK(false, "Slice must have either 4 or 5 inputs.");
             }

             if (start.get_partial_shape().rank().is_dynamic() || start.get_partial_shape().rank().get_length() == 0) {
                 start = context.mark_node(std::make_shared<opset8::Unsqueeze>(start, axis_0));
             }
             if (end.get_partial_shape().rank().is_dynamic() || end.get_partial_shape().rank().get_length() == 0) {
                 end = context.mark_node(std::make_shared<opset8::Unsqueeze>(end, axis_0));
             }
             if (step.get_partial_shape().rank().is_dynamic() || step.get_partial_shape().rank().get_length() == 0) {
                 step = context.mark_node(std::make_shared<opset8::Unsqueeze>(step, axis_0));
             }
             return {context.mark_node(std::make_shared<opset8::Slice>(context.get_input(0), start, end, step, dim))};
         }},

        {"prim::Loop", op::translate_loop},
        {"prim::If", op::translate_if},

        {"prim::Constant",
         [](NodeContext& context) -> OutputVector {
             return context.as_constant();
         }},

        {"aten::dim",
         [](NodeContext& context) -> OutputVector {
             auto shape = std::make_shared<opset8::ShapeOf>(context.get_input(0), element::i32);
             auto rank = std::make_shared<opset8::ShapeOf>(shape, element::i32);
             auto squeeze = std::make_shared<opset8::Squeeze>(rank);
             return {context.mark_node(squeeze)};
         }},

        {"aten::reciprocal",
         [](NodeContext& context) -> OutputVector {
             auto x = context.get_input(0);
             auto const_neg_1 = opset8::Constant::create(element::i32, Shape{}, {-1});
             auto cast = std::make_shared<opset8::ConvertLike>(const_neg_1, x);
             auto power = std::make_shared<opset8::Power>(x, cast);
             return {context.mark_node(power)};
         }},

        {"aten::sub",
         [](NodeContext& context) -> OutputVector {
             auto x = context.get_input(0);
             auto y = context.get_input(1);
             // default is 1 so no need to multiply by alpha
             if (!context.input_is_none(2)) {
                 auto alpha = context.get_input(2);
                 auto casted_alpha = std::make_shared<opset8::ConvertLike>(alpha, y);
                 y = std::make_shared<opset8::Multiply>(casted_alpha, y);
             }
             return {context.mark_node(std::make_shared<opset8::Subtract>(x, y))};
         }},

        {"aten::eq",
         [](NodeContext& context) -> OutputVector {
             auto x = context.get_input(0);
             auto y = context.get_input(1);
             return {context.mark_node(std::make_shared<opset8::Equal>(x, y))};
         }},

        {"aten::ne",
         [](NodeContext& context) -> OutputVector {
             auto x = context.get_input(0);
             auto y = context.get_input(1);
             return {context.mark_node(std::make_shared<opset8::NotEqual>(x, y))};
         }},

        {"aten::gt",
         [](NodeContext& context) -> OutputVector {
             auto x = context.get_input(0);
             auto y = context.get_input(1);
             return {context.mark_node(std::make_shared<opset8::Greater>(x, y))};
         }},

        {"aten::lt",
         [](NodeContext& context) -> OutputVector {
             auto x = context.get_input(0);
             auto y = context.get_input(1);
             return {context.mark_node(std::make_shared<opset8::Less>(x, y))};
         }},

        {"aten::neg",
         [](NodeContext& context) -> OutputVector {
             auto x = context.get_input(0);
             auto const_neg_1 = opset8::Constant::create(element::i32, Shape{}, {-1});
             auto cast = std::make_shared<opset8::ConvertLike>(const_neg_1, x);
             return {context.mark_node(std::make_shared<opset8::Multiply>(x, cast))};
         }},

        // TODO: Don't know how to change it quickly to be compiled, consult with Maxim
        /*{ "prim::ConstantChunk", [&]() -> OutputVector {
            auto chunks = node->i(attr::chunks); // FIXME: create actual attribute function
            auto dim = node->i(attr::dim);
            auto dim_const = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {dim}));
            auto split = context.mark_node(std::make_shared<opset8::Split>(context.get_input(0), dim_const, chunks));
            return split->outputs();
        }},*/

    };
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
