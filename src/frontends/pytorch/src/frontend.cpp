#include "openvino/frontend/pytorch/frontend.hpp"

#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <string>

#include "exception.hpp"
#include "input_model.hpp"
#include "node_context.hpp"
#include "openvino/frontend/exception.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
int NUMBER = 0;
int COUNTER = 0;
int LEVEL = 0;

std::shared_ptr<ov::Model> convert_pytorch_model(std::shared_ptr<Decoder> pytorch_model);

const auto relu = [](NodeContext& context) -> OutputVector {
    return {context.mark_node(std::make_shared<opset7::Relu>(context.get_input(0)))};
};
const auto add = [](NodeContext& context) -> OutputVector {
    return {context.mark_node(std::make_shared<opset7::Add>(context.get_input(0), context.get_input(1)))};
};

const std::map<std::string, std::function<OutputVector(NodeContext& context)>> CONVERTERS_MAP = {

    {"aten::relu", relu},
    {"aten::relu_", relu},  // inplace version of relu, for us it is identical to regular relu

    {"aten::conv2d",
     [](NodeContext& context) -> OutputVector {
         auto strides = context.const_input<Strides>(3);
         auto pads_begin = context.const_input<CoordinateDiff>(4);  // FIXME: The same 4 is used twice
         auto pads_end = context.const_input<CoordinateDiff>(4);    // FIXME: The same 4 is used twice
         auto dilations = context.const_input<Strides>(5);
         auto groups = context.const_input<int64_t>(6);

         std::shared_ptr<ov::Node> conv;
         if (groups == 1) {
             conv = std::make_shared<opset7::Convolution>(context.get_input(0),
                                                          context.get_input(1),
                                                          strides,
                                                          pads_begin,
                                                          pads_end,
                                                          dilations);
         } else {
             conv = std::make_shared<opset7::GroupConvolution>(
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
             conv = std::make_shared<opset7::Convolution>(context.get_input(0),
                                                          context.get_input(1),
                                                          strides,
                                                          pads_begin,
                                                          pads_end,
                                                          dilations);
         } else {
             conv = std::make_shared<opset7::GroupConvolution>(
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
         return {
             context.mark_node(std::make_shared<opset7::BatchNormInference>(context.get_input(0),
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
         auto axes =
             context.mark_node(opset7::Constant::create(element::i64, Shape{1}, {-1}));  // TODO: support any dimention
         auto mvn = context.mark_node(
             std::make_shared<opset7::MVN>(context.get_input(0), axes, true, eps, op::MVNEpsMode::INSIDE_SQRT));
         std::shared_ptr<ov::Node> out_node = std::dynamic_pointer_cast<ov::Node>(mvn);
         if (!context.input_is_none(2)) {
             auto mul = std::make_shared<opset7::Multiply>(out_node, context.get_input(2));
             out_node = std::dynamic_pointer_cast<ov::Node>(mul);
         }
         if (!context.input_is_none(3)) {
             auto add = std::make_shared<opset7::Add>(out_node, context.get_input(3));
             out_node = std::dynamic_pointer_cast<ov::Node>(add);
         }
         return {context.mark_node(out_node)};
     }},

    {"aten::add", add},
    {"aten::add_", add},  // inplace version of add, for us it is identical to regular add

    {"aten::mul",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::Multiply>(context.get_input(0), context.get_input(1)))};
     }},

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
             std::make_shared<opset7::Divide>(context.get_input(0), context.get_input(1), pythondiv))};
     }},

    {"aten::tanh",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::Tanh>(context.get_input(0)))};
     }},

    {"aten::elu",
     [](NodeContext& context) -> OutputVector {
         auto alpha = context.const_input<float>(1);
         return {context.mark_node(std::make_shared<opset7::Elu>(context.get_input(0), alpha))};
     }},

    {"aten::sigmoid",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::Sigmoid>(context.get_input(0)))};
     }},

    {"aten::gelu",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::Gelu>(context.get_input(0)))};
     }},

    {"aten::sqrt",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::Sqrt>(context.get_input(0)))};
     }},

    {"aten::abs",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::Abs>(context.get_input(0)))};
     }},

    {"aten::square",
     [](NodeContext& context) -> OutputVector {
         auto input_0 = context.get_input(0);
         auto const_2 = context.mark_node(opset7::Constant::create(input_0.get_element_type(), Shape{1}, {2}));
         return {context.mark_node(std::make_shared<opset7::Power>(input_0, const_2))};
     }},

    {"aten::hardtanh",
     [](NodeContext& context) -> OutputVector {
         auto min = context.const_input<float>(1);
         auto max = context.const_input<float>(2);
         return {context.mark_node(std::make_shared<opset7::Clamp>(context.get_input(0), min, max))};
     }},

    {"aten::hardsigmoid",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::HSigmoid>(context.get_input(0)))};
     }},

    {"aten::hardswish",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::HSwish>(context.get_input(0)))};
     }},

    {"aten::relu6",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::Clamp>(context.get_input(0), 0., 6.))};
     }},

    {"aten::softmax",
     [](NodeContext& context) -> OutputVector {
         auto axis = context.const_input<int64_t>(1);
         if (axis < 0) {
             auto in_rank = context.get_input(0).get_partial_shape().rank();
             OV_FRONTEND_REQUIRE(in_rank.is_static());
             axis = in_rank.get_length() + axis;
         }
         return {context.mark_node(std::make_shared<opset7::Softmax>(context.get_input(0), static_cast<size_t>(axis)))};
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
         OV_FRONTEND_REQUIRE(listConstruct_fw_node->get_decoder()->get_op_type() == "prim::ListConstruct");
         auto axis = context.const_input<int64_t>(1);
         OutputVector inputs;
         for (auto& input : listConstruct->inputs()) {
             inputs.push_back(input.get_source_output());
         }
         auto result = context.mark_node(std::make_shared<opset7::Concat>(inputs, axis));
         auto list_set = listConstruct_fw_node->get_rt_info()["pt_node"].as<std::set<const Node*>>();
         result->get_rt_info()["pt_node"].as<std::set<const Node*>>().insert(list_set.begin(), list_set.end());
         return {result};
     }},

    {"aten::matmul",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::MatMul>(context.get_input(0), context.get_input(1)))};
     }},

    {"aten::mm",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::MatMul>(context.get_input(0), context.get_input(1)))};
     }},

    {"aten::linear",
     [](NodeContext& context) -> OutputVector {
         auto matmul = std::make_shared<opset7::MatMul>(context.get_input(0), context.get_input(1), false, true);
         return {context.mark_output(make_optional_bias(matmul, context, 2))};
     }},

    {"aten::max_pool2d",
     [](NodeContext& context) -> OutputVector {
         auto kernel = context.const_input<Shape>(1);
         auto strides = context.const_input<Strides>(2);
         auto pads_begin = context.const_input<Shape>(3);  // FIXME: The same 3 is used twice
         auto pads_end = context.const_input<Shape>(3);    // FIXME: The same 3 is used twice
         auto dilations = context.const_input<Strides>(4);
         auto rounding_type = context.const_input<bool>(5) ? op::RoundingType::CEIL : op::RoundingType::FLOOR;

         // TODO: Upgrade to opset8::MaxPool to use dilations; for now we suppose they are all zeros
         return {context.mark_node(std::make_shared<opset7::MaxPool>(context.get_input(0),
                                                                     strides,
                                                                     /*dilations,*/ pads_begin,
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
         auto rounding_type = context.const_input<bool>(4) ? op::RoundingType::CEIL : op::RoundingType::FLOOR;
         auto exclude_pad = !context.const_input<bool>(5);
         // TODO: support divisor override
         // auto divisor_override = context.const_input<int64_t>(6);

         return {context.mark_node(std::make_shared<opset7::AvgPool>(context.get_input(0),
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     kernel,
                                                                     exclude_pad,
                                                                     rounding_type))};
     }},

    {"aten::adaptive_avg_pool2d",
     [](NodeContext& context) -> OutputVector {
         return {
             context.mark_node(std::make_shared<opset8::AdaptiveAvgPool>(context.get_input(0), context.get_input(1)))};
     }},

    {"aten::adaptive_max_pool2d",
     [](NodeContext& context) -> OutputVector {
         auto adaptive_max_pool =
             context.mark_node(std::make_shared<opset8::AdaptiveMaxPool>(context.get_input(0), context.get_input(1)));
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
             context.mark_node(opset7::Constant::create(element::i64, {new_shape.size()}, new_shape));
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

    {"aten::as_tensor",
     [](NodeContext& context) -> OutputVector {
         OV_FRONTEND_REQUIRE(context.const_input<int64_t>(1) == 6);
         OV_FRONTEND_REQUIRE(context.input_is_none(2));
         // auto new_shape_const = context.mark_node(opset7::Constant::create(element::i64, {1}, {1}));
         // return { context.mark_node(std::make_shared<opset8::Reshape>(context.get_input(0),
         // new_shape_const->output(0), true)) };
         return {context.mark_output(context.get_input(0))};
     }},

    {"aten::Int",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset8::Convert>(context.get_input(0), element::i64))};
     }},

    {"aten::to",
     [](NodeContext& context) -> OutputVector {
         auto dtype = element::f32;
         // TODO: figure out all inputs meaning
         OV_FRONTEND_REQUIRE(context.const_input<int64_t>(1) == 6);
         OV_FRONTEND_REQUIRE(context.const_input<bool>(2) == false);
         OV_FRONTEND_REQUIRE(context.const_input<bool>(3) == false);
         OV_FRONTEND_REQUIRE(context.input_is_none(4));
         return {context.mark_node(std::make_shared<opset8::Convert>(context.get_input(0), dtype))};
     }},

    {"aten::permute",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset7::Transpose>(context.get_input(0), context.get_input(1)))};
     }},

    {"aten::embedding",
     [](NodeContext& context) -> OutputVector {
         // TODO: find out the meaning of input idx 2
         OV_FRONTEND_REQUIRE(context.const_input<bool>(3) == false);
         OV_FRONTEND_REQUIRE(context.const_input<bool>(4) == false);
         auto axis_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
         return {
             context.mark_node(std::make_shared<opset7::Gather>(context.get_input(0), context.get_input(1), axis_0))};
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
         auto order_const = context.mark_node(opset7::Constant::create(element::i64, {order.size()}, order));
         return {context.mark_node(std::make_shared<opset7::Transpose>(context.get_input(0), order_const))};
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
                     context.mark_node(std::make_shared<opset7::Unsqueeze>(input.get_source_output(), axis_0));
                 inputs.push_back(unsqueeze);
             }
             auto concat = context.mark_node(std::make_shared<opset7::Concat>(inputs, 0));
             reshape = context.mark_node(std::make_shared<opset7::Reshape>(context.get_input(0), concat, false));
             auto list_set = shape_node_fw_node->get_rt_info()["pt_node"].as<std::set<const Node*>>();
             reshape->get_rt_info()["pt_node"].as<std::set<const Node*>>().insert(list_set.begin(), list_set.end());
         } else {
             reshape = context.mark_node(
                 std::make_shared<opset7::Reshape>(context.get_input(0), context.get_input(1), false));
         }
         return {reshape};
     }},

    {"aten::unsqueeze",
     [](NodeContext& context) -> OutputVector {
         return {context.mark_node(std::make_shared<opset8::Unsqueeze>(context.get_input(0), context.get_input(1)))};
     }},

    {"aten::rsub",
     [](NodeContext& context) -> OutputVector {
         // reverse aten::sub other - self * alpha
         auto alpha_casted = context.mark_node(
             std::make_shared<opset8::Convert>(context.get_input(2), context.get_input(0).get_element_type()));
         auto alpha_mul = context.mark_node(std::make_shared<opset8::Multiply>(context.get_input(0), alpha_casted));
         return {context.mark_node(std::make_shared<opset8::Subtract>(context.get_input(1), alpha_mul))};
     }},

    // temporarily disable slice
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

    /* TODO: Don't know how to change it quickly to be compiled, consult with Maxim
    { "prim::ConstantChunk", [&]() -> OutputVector {
        auto chunks = node->i(attr::chunks); // FIXME: create actual attribute function
        auto dim = node->i(attr::dim);
        auto dim_const = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {dim}));
        auto split = context.mark_node(std::make_shared<opset8::Split>(context.get_input(0), dim_const, chunks));
        return split->outputs();
    }},
    */

    {"prim::Loop",
     [](NodeContext& context) -> OutputVector {
         auto loop = std::make_shared<opset8::Loop>(context.get_input(0), context.get_input(1));
         auto decoder = context.get_decoder();
         OV_FRONTEND_REQUIRE(decoder->get_subgraph_size() == 1);
         auto subgraph_decoder = decoder->get_subgraph_decoder(0);
         auto body = convert_pytorch_model(subgraph_decoder);
         loop->set_function(body);
         opset8::Loop::SpecialBodyPorts spec_ports{0, 0};
         loop->set_special_body_ports(spec_ports);

         auto inputs = subgraph_decoder->inputs();
         std::set<size_t> input_idxs(inputs.begin(), inputs.end());
         std::map<size_t, ParameterVector> inputs_map;
         std::map<size_t, std::shared_ptr<opset8::Result>> outputs_map;

         auto body_parameters = body->get_parameters();
         // #0 parameter is counter
         for (int i = 1; i < body_parameters.size(); i++) {
             auto param = body_parameters[i];
             auto name = param->get_output_tensor(0).get_any_name();
             size_t input_idx = (size_t)std::stoll(name);
             if (inputs_map.count(input_idx)) {
                 inputs_map[input_idx] = {param};
             } else {
                 inputs_map[input_idx].push_back(param);
             }
         }
         // TODO: Connect back edges, and out condition
         for (auto result : body->get_results()) {
             auto name = result->input(0).get_tensor().get_any_name();
             size_t out_idx = (size_t)std::stoll(name);
             FRONT_END_OP_CONVERSION_CHECK(outputs_map.count(out_idx) == 0,
                                           "More then one body output with same tensor name.");
             outputs_map[out_idx] = result;
         }
         for (auto input : inputs_map) {
             if (!input_idxs.count(input.first)) {
                 auto external_output = context.get_tensor_from_ext_or_create_ext_input(input.first);
                 loop->set_invariant_inputs(external_output, input.second);
             } else {
                 auto external_output = context.get_tensor_from_ext(input.first);
                 if (external_output.get_node()) {
                     loop->set_invariant_inputs(external_output, input.second);
                 }
             }
         }
         for (auto output : outputs_map) {
             context.add_tensor_to_external_context(output.first, loop->set_body_outputs({output.second}));
         }
         loop->validate_and_infer_types();
         return {context.mark_node(loop)->outputs()};
     }},

    {"prim::If",
     [](NodeContext& context) -> OutputVector {
         auto if_node = std::make_shared<opset8::If>(context.get_input(0));
         context.mark_node(if_node);
         auto decoder = context.get_decoder();
         OV_FRONTEND_REQUIRE(decoder->get_subgraph_size() == 2);

         auto then_decoder = decoder->get_subgraph_decoder(0);
         auto then_body = convert_pytorch_model(then_decoder);
         if_node->set_then_body(then_body);
         auto then_inputs = then_decoder->inputs();

         auto else_decoder = decoder->get_subgraph_decoder(1);
         auto else_body = convert_pytorch_model(else_decoder);
         if_node->set_else_body(else_body);
         auto else_inputs = else_decoder->inputs();

         std::set<size_t> input_idxs;
         input_idxs.insert(then_inputs.begin(), then_inputs.end());
         input_idxs.insert(else_inputs.begin(), else_inputs.end());

         std::map<size_t, ParameterVector> inputs_map;
         std::map<size_t, ResultVector> outputs_map;
         for (auto param : then_body->get_parameters()) {
             auto name = param->get_output_tensor(0).get_any_name();
             size_t input_idx = (size_t)std::stoll(name);
             FRONT_END_OP_CONVERSION_CHECK(inputs_map.count(input_idx) == 0,
                                           "More then one then_body input with same tensor name: ",
                                           inputs_map.at(input_idx)[0],
                                           " adding: ",
                                           param);
             inputs_map[input_idx] = {param, nullptr};
         }
         for (auto param : else_body->get_parameters()) {
             auto name = param->get_output_tensor(0).get_any_name();
             size_t input_idx = (size_t)std::stoll(name);
             if (inputs_map.count(input_idx)) {
                 inputs_map[input_idx][1] = param;
             } else {
                 inputs_map[input_idx] = {nullptr, param};
             }
         }
         std::map<size_t, std::shared_ptr<opset8::Result>> then_body_results;
         std::map<size_t, std::shared_ptr<opset8::Result>> else_body_results;
         std::set<size_t> output_idxs;
         for (auto result : then_body->get_results()) {
             auto name = result->input(0).get_tensor().get_any_name();
             size_t output_idx = (size_t)std::stoll(name);
             FRONT_END_OP_CONVERSION_CHECK(then_body_results.count(output_idx) == 0,
                                           "More then one then_body output with same tensor name: ",
                                           then_body_results.at(output_idx),
                                           " adding: ",
                                           result);
             then_body_results[output_idx] = result;
             output_idxs.insert(output_idx);
         }
         for (auto result : else_body->get_results()) {
             auto name = result->input(0).get_tensor().get_any_name();
             size_t output_idx = (size_t)std::stoll(name);
             FRONT_END_OP_CONVERSION_CHECK(else_body_results.count(output_idx) == 0,
                                           "More then one then_body output with same tensor name: ",
                                           else_body_results.at(output_idx),
                                           " adding: ",
                                           result);
             then_body_results[output_idx] = result;
             output_idxs.insert(output_idx);
         }
         OutputVector res;
         for (int i = 0; i < context.num_of_outputs(); i++) {
             res.push_back(if_node->set_output(then_body->get_results()[i], else_body->get_results()[i]));
             OV_FRONTEND_REQUIRE(output_idxs.erase(then_decoder->output(i)));
             OV_FRONTEND_REQUIRE(output_idxs.erase(else_decoder->output(i)));
         }
         for (auto output_idx : output_idxs) {
             if (!then_body_results.count(output_idx)) {
                 // Need to add Parameter->Result construction in then body
                 auto new_parameter = std::make_shared<opset8::Parameter>(element::dynamic, PartialShape::dynamic());
                 new_parameter->get_output_tensor(0).add_names({std::to_string(output_idx)});
                 auto new_result = std::make_shared<opset8::Result>(new_parameter);
                 then_body->add_parameters({new_parameter});
                 then_body->add_results({new_result});
                 then_body->validate_nodes_and_infer_types();
                 FRONT_END_OP_CONVERSION_CHECK(inputs_map.count(output_idx), "Input must exist in else body");
                 inputs_map[output_idx][0] = new_parameter;
                 then_body_results[output_idx] = new_result;
                 std::cout << "[ WARNING ] Modified then body: " << if_node << std::endl;
             } else if (!else_body_results.count(output_idx)) {
                 // Need to add Parameter->Result construction in else body
                 auto new_parameter = std::make_shared<opset8::Parameter>(element::dynamic, PartialShape::dynamic());
                 new_parameter->get_output_tensor(0).add_names({std::to_string(output_idx)});
                 auto new_result = std::make_shared<opset8::Result>(new_parameter);
                 else_body->add_parameters({new_parameter});
                 else_body->add_results({new_result});
                 else_body->validate_nodes_and_infer_types();
                 FRONT_END_OP_CONVERSION_CHECK(inputs_map.count(output_idx), "Input must exist in then body");
                 inputs_map[output_idx][1] = new_parameter;
                 else_body_results[output_idx] = new_result;
                 std::cout << "[ WARNING ] Modified else body: " << if_node << std::endl;
             }
         }
         // Create prim::If inputs and outputs
         for (auto input : inputs_map) {
             if (!input_idxs.count(input.first)) {
                 auto external_output = context.get_tensor_from_ext_or_create_ext_input(input.first);
                 if_node->set_input(external_output, input.second[0], input.second[1]);
             } else {
                 auto external_output = context.get_tensor_from_ext(input.first);
                 if (external_output.get_node()) {
                     if_node->set_input(external_output, input.second[0], input.second[1]);
                 }
             }
         }
         for (auto output_idx : output_idxs) {
             context.add_tensor_to_external_context(
                 output_idx,
                 if_node->set_output(then_body_results.at(output_idx), else_body_results.at(output_idx)));
         }
         if_node->validate_and_infer_types();
         return res;
     }},

    {"prim::Constant",
     [](NodeContext& context) -> OutputVector {
         return context.as_constant();
     }},

    /*{"aten::append",
     [](NodeContext& context) -> OutputVector {
        // schema: aten::append.t(t[](a!) self, t(c -> *) el) -> (t[](a!))
        OutputVector res{context.mark_node(std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs(),
    context.get_decoder()->num_of_outputs()))};
        // append writes to input 0, so we need to replace this input with output from append
        context.mutate_input(0, res[0]);
        return res;
     }},

    {"aten::update",
     [](NodeContext& context) -> OutputVector {
        // schema: aten::update.str(Dict(str, t)(a!) self, Dict(str, t)(a!) to_add) -> ()
        // pt node has no outputs, need to add one
        auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs(),
    context.get_decoder()->num_of_outputs() + 1); OutputVector res{context.mark_node(fw_node)->output(0)};
        // update writes to input 0, so we need to replace this input with output from update
        context.mutate_input(0, res[0]);
        return {};
     }}*/

};

OutputVector convert_node(NodeContext* context) {
    // std::cout << "[  ----  DEBUG  ---- ] convert_node\n";

    // std::cerr << "---\nAttempting to convert " << node->kind().toQualString() << "\n";
    // node->dump();

    // std::cerr << "[ DEBUG ] Attempting to convert " << context.get_op_type() << "\n";

    try {
        auto it = CONVERTERS_MAP.find(context->get_op_type());
        if (it != CONVERTERS_MAP.end()) {
            // std::cout << "FOUND converter for " << context.get_op_type() << "\n";
            return it->second(*context);
        } else {
            std::cout << "DIDN'T FIND converter for " << context->get_op_type() << "\n";
        }

    }
    // catch(pybind11::error_already_set& e) {
    //     std::cout << "Python exception: " << e << "\n";
    // }
    catch (std::runtime_error& e) {
        std::cout << "Exception happened during conversion of op: " << context->get_op_type() << ": " << e.what() << '\n';
        // throw;
    } catch (...) {
        std::cout << "Some exception happened during conversion of node of type: " << context->get_op_type()
                  << std::endl;
        // throw;
    }
    // if (node->kind() != prim::ListConstruct) {
    //     std::cout << "Making unsupported " << node->kind().toQualString() << std::endl;
    //     node->dump();
    // }

    // Create PtFrameworkNode for everything that wasn't able to be converted normally
    // Pay attention to subgraphs that may appear in the node
    // std::cerr << "[ DEBUG ] Before PtFramewokNode creation\n";

    auto schema = context->get_schema();
    if (schema.find('!') != std::string::npos) {
        // Hack. Can indicate mutable inputs, but can it be reliable?
        auto fw_node = std::make_shared<PtFrameworkNode>(context->get_decoder(),
                                                         context->inputs(),
                                                         context->get_decoder()->num_of_outputs() + 1);
        fw_node->set_friendly_name(context->get_op_type() + ":" + std::to_string(COUNTER++));
        auto outputs = fw_node->outputs();
        // update writes to input 0, so we need to replace this input with output from update
        context->mutate_input(0, outputs.back());
        std::cerr << "[ WARNING ] Created node with mutated 0 input. Schema: " << schema << std::endl;
        context->get_decoder()->mark_node(fw_node);
        return outputs;
    }
    auto fw_node = std::make_shared<PtFrameworkNode>(context->get_decoder(),
                                                     context->inputs(),
                                                     context->get_decoder()->num_of_outputs());
    fw_node->set_friendly_name(context->get_op_type() + ":" + std::to_string(COUNTER++));

    std::map<size_t, ParameterVector> inputs_map;
    std::map<size_t, ResultVector> outputs_map;
    std::set<size_t> input_idxs;
    for (size_t i = 0; i < context->get_decoder()->get_subgraph_size(); ++i) {
        auto subgraph_decoder = context->get_decoder()->get_subgraph_decoder(i);
        auto inputs = subgraph_decoder->inputs();
        input_idxs.insert(inputs.begin(), inputs.end());
        auto body = convert_pytorch_model(subgraph_decoder);
        fw_node->set_function(i, body);
        for (auto param : body->get_parameters()) {
            auto name = param->get_output_tensor(0).get_any_name();
            size_t input_idx = (size_t)std::stoll(name);
            inputs_map[input_idx].push_back(param);
        }
        for (auto result : body->get_results()) {
            auto name = result->input(0).get_tensor().get_any_name();
            size_t out_idx = (size_t)std::stoll(name);
            FRONT_END_OP_CONVERSION_CHECK(outputs_map.count(out_idx) == 0,
                                          "More then one body output with same tensor name.");
            outputs_map[out_idx].push_back(result);
        }
    }
    for (auto input : inputs_map) {
        if (!input_idxs.count(input.first)) {
            auto external_output = context->get_tensor_from_ext_or_create_ext_input(input.first);
            fw_node->set_invariant_inputs(external_output, input.second);
        } else {
            auto external_output = context->get_tensor_from_ext(input.first);
            if (external_output.get_node()) {
                fw_node->set_invariant_inputs(external_output, input.second);
            }
        }
    }
    for (auto output : outputs_map) {
        context->add_tensor_to_external_context(output.first, fw_node->set_body_outputs(output.second));
    }
    return context->get_decoder()->mark_node(fw_node)->outputs();
}

std::shared_ptr<ov::Model> convert_pytorch_model(std::shared_ptr<Decoder> pytorch_model) {
    LEVEL++;
    // std::cout << "=====Convert model:" << LEVEL << " start=====" << std::endl;
    std::shared_ptr<ov::Model> resulting_model;  // define here to make a conversion in a nested scope
    {
        ParameterVector parameters;
        TensorMap tensor_map;  // tensor map of the current context
        std::set<size_t> mutated_tensors;

        //  Go over all pytorch_model inputs and register them in the tensor map:
        auto inputs = pytorch_model->inputs();
        // std::cout << "[  ---  DEBUG --- ] convert_pytorch_model: number of inputs: " << inputs.size() << '\n';
        for (int i = 0; i < inputs.size(); ++i) {
            // std::cout << "Input: " << i << ": " << inputs[i] << "\n";
            PartialShape ps = pytorch_model->get_input_shape(i);
            // std::cout << "PartialShape = " << ps << "\n";
            auto parameter =
                std::make_shared<opset7::Parameter>(ov::element::custom, pytorch_model->get_input_type(i), ps);
            parameter->get_output_tensor(0).add_names({std::to_string(pytorch_model->input(i))});
            // std::cout << "Parameter: " << parameter << "\n";
            parameters.push_back(parameter);
            auto order = pytorch_model->get_input_transpose_order(i);
            if (order.size() > 0 && !std::is_sorted(order.begin(), order.end())) {
                OV_FRONTEND_REQUIRE(ps.is_static());  // TODO: make dynamic
                auto sh = ps.get_shape();
                Shape new_shape(sh.size());
                for (int i = 0; i < sh.size(); i++) {
                    new_shape[order[i]] = sh[i];
                }
                auto shape_const = opset7::Constant::create(element::i64, {new_shape.size()}, new_shape);
                auto reshape = std::make_shared<opset7::Reshape>(parameter, shape_const, false);
                auto order_const = opset7::Constant::create(element::i32, {order.size()}, order);
                auto transpose = std::make_shared<opset7::Transpose>(reshape, order_const);
                tensor_map[pytorch_model->input(i)] = transpose;
            } else {
                tensor_map[pytorch_model->input(i)] = parameter;
            }
            // std::cout << "Level:" << LEVEL << " Added model input: " << tensor_map[pytorch_model->input(i)] << std::endl;
        }

        auto node_visitor = [&](std::shared_ptr<Decoder> node) {
            // std::cerr << "Node convert start" << std::endl;

            // Even if node has no outputs it can mutate input. Remove this
            /*if (!node->num_of_outputs()) {
                bool no_subgraph_outputs = true;
                for (int i = 0; i < node->get_subgraph_size(); i++) {
                    auto subgraph = node->get_subgraph_decoder(i);
                    if (subgraph->num_of_outputs() > 0) {
                        no_subgraph_outputs = false;
                    }
                }
                // TODO: remove this check
                if (no_subgraph_outputs && node->get_schema().find("!") == std::string::npos) {
                    std::cout << "Node has no outputs: " << node->get_op_type() << " Skipping." << std::endl;
                    return;
                }
            }*/

            // Explore all inputs of node. Node may refer to input value that hasn't been created in the current scope.
            // But this value can be found in the outer scope, for this purpose we need to search node in
            // external_tensor_map as well

            auto raw_inputs = node->inputs();
            for (size_t i = 0; i < raw_inputs.size(); ++i) {
                auto input = node->input(i);
                if (tensor_map.find(input) == tensor_map.end()) {
                    std::cout << "Level:" << LEVEL << " Trampoline for input index " << i << " with value " << input
                              << "\n";
                    //  input refers value in the outer scope, need to create a new Parameter in the current scope
                    //  TODO: Connect outer scope and inner scope properly -- should be handled at the level of that
                    //  operation that introduced this nest of scopes (e.g. loop or if)
                    //  TODO: Eliminate duplication with the main code for Parameters creation
                    //  TODO: There is no real search for values in outer scope because we don't need to link the usage
                    //  and definition together at this point -- need to do that otherwise graph will fall apart
                    PartialShape ps = node->get_input_shape(i);
                    auto parameter = std::make_shared<opset7::Parameter>(node->get_input_type(i), ps);
                    // TODO: Missing get_input_transpose_order handling for not trivial layouts
                    tensor_map[input] = parameter;
                    // std::cout << "Parameter created\n";
                    // set name of parameter to the index of node in the model
                    parameter->get_output_tensor(0).add_names({std::to_string(input)});
                    parameters.push_back(parameter);
                    // std::cout << "External tensor: " << input << " node: " << external_tensor_map.at(input) <<
                    // std::endl;
                }
            }
            // std::cerr << "Node convert before translator: " << node->get_op_type() << ", schema: " <<
            // node->get_schema() << std::endl;

            auto context = NodeContext(node, &tensor_map, &parameters);
            auto converted_outputs = convert_node(&context);
            // std::cerr << "Node convert before outputs" << std::endl;

            auto mutated_t = context.get_mutated_tensors();
            mutated_tensors.insert(mutated_t.begin(), mutated_t.end());

            auto fw_outputs = node->outputs();
            // ops with subgraphs has more outputs
            FRONT_END_OP_CONVERSION_CHECK(fw_outputs.size() <= converted_outputs.size(),
                                          "Number of ",
                                          node->get_op_type(),
                                          " outputs greater then number of converted outputs.");

            // TODO: Make sure that mapping of fw_outputs to converted_outputs does always work
            // FIXME: Now it is not true for at least prim::Constant
            for (size_t i = 0; i < fw_outputs.size(); ++i) {
                size_t fw_tensor_id = node->output(i);
                if (tensor_map.find(fw_tensor_id) != tensor_map.end()) {
                    // std::cerr << "Duplicated producer for tensor with id = " << fw_tensor_id << " discovered at
                    // output "
                    //     << "port " << i << " of node " << node->kind().toQualString() << "\n";
                    throw std::runtime_error("Duplicated producer for PT value with unique ID: " +
                                             std::to_string(fw_tensor_id));
                }

                // Output shape of converted node should match the original output shape
                // std::cerr << "[ DEBUG ] PT output shape = " << get_ov_shape(fw_outputs[i]) << '\n';
                // std::cerr << "[ DEBUG ] OV output shape = " << converted_outputs[i].get_partial_shape() << '\n';
                // OV_FRONTEND_REQUIRE(get_ov_shape(fw_outputs[i]) == converted_outputs[i].get_partial_shape());

                tensor_map[fw_tensor_id] = converted_outputs[i];
                converted_outputs[i].get_tensor().add_names({std::to_string(fw_tensor_id)});
                // std::cout << "Level:" << LEVEL << " Added node: " << converted_outputs[i] << std::endl;
                //  std::cout << "Converted node output " << fw_tensor_id << ": " << converted_outputs[i] << std::endl;
            }
            // std::cout << "Node convert end" << std::endl;
        };

        OV_FRONTEND_REQUIRE(pytorch_model->get_subgraph_size() == 1);
        pytorch_model->visit_subgraph(0, node_visitor);
        // std::cout << "All nodes convert end" << std::endl;

        ResultVector results;
        // std::cerr << "Outputs:" << pytorch_model->num_of_outputs() << "\n";
        for (size_t i = 0; i < pytorch_model->num_of_outputs(); ++i) {
            size_t id = pytorch_model->output(i);
            // std::cerr << "Output:" << i << ": " << id << "\n";
            // std::cout << "value = " << id << '\n';
            // std::cout << "X\n";
            if (tensor_map.find(id) == tensor_map.end()) {
                // Not found in this scope, searching in the outer scope
                // TODO: do real search here, skipped for now

                auto parameter = std::make_shared<opset7::Parameter>(element::dynamic, PartialShape::dynamic());
                parameter->get_output_tensor(0).add_names({std::to_string(id)});
                parameters.push_back(parameter);
                tensor_map[id] = parameter;
                std::cout << "Level:" << LEVEL << "Added new parameter based on external value " << id << "\n";
            }
            auto ov_output = tensor_map[id];
            // std::cout << "X\n";
            auto order = pytorch_model->get_output_transpose_order(i);
            // std::cout << "X\n";
            if (order.size() > 0 && !std::is_sorted(order.begin(), order.end())) {
                throw "Output strides have wrong order.";
            }
            // TODO: remove when all nodes has ids
            ov_output.add_names({std::to_string(id)});
            // std::cout << "X\n";
            // std::cout << ov_output << '\n';
            auto result = std::make_shared<opset7::Result>(ov_output);
            // std::cout << "X\n";
            results.push_back(result);
            // std::cerr << "Model result " << result << "\n";
        }

        // Since parameters can be added we need to list all current parameters
        std::set<size_t> param_names;
        for (auto param : parameters) {
            auto name = param->get_output_tensor(0).get_any_name();
            size_t input_idx = (size_t)std::stoll(name);
            param_names.insert(input_idx);
        }
        for (auto tensor : mutated_tensors) {
            if (param_names.count(tensor)) {
                OV_FRONTEND_REQUIRE(tensor_map.count(tensor));
                // model input was mutated we need to make a result for it
                results.push_back(std::make_shared<opset7::Result>(tensor_map.at(tensor)));
            }
        }
        // std::cout << "Y\n";

        /*for (size_t i = 0; i < parameters.size(); ++i) {
            auto parameter = parameters[i];
            // std::cerr << "parameter[" << i << "].shape = "
            //     << parameter->get_output_shape(0) << ", consumers: " <<
            //     parameter->output(0).get_target_inputs().size() << "\n";
        }*/
        // std::cout << "Convert end" << std::endl;
        // std::cout << "Number of values collected: " << tensor_map.size() << "\n";

        // std::cout << "=====Construct model start=====" << std::endl;
        /*std::cout << "=====Tensor map start=====" << std::endl;
        for (auto node : tensor_map) {
            std::cout << node.first << ": " << node.second.get_node_shared_ptr() << std::endl;
        }*/
        resulting_model = std::make_shared<ov::Model>(results, parameters);
        /*std::string m_name = "model_" + std::to_string(LEVEL) + "_" + std::to_string(NUMBER++);
        try {
            ov::serialize(resulting_model, m_name + ".xml", m_name + ".bin");
        } catch (...) {
            std::cout << "Exception happened during model serialization: " + m_name << std::endl;
        }*/
        // std::cout << "=====Construct model end=====" << std::endl;

        // Did a conversion in a nested scope to automatically remove any holders of nodes except those in the graph
    }

    // std::cout << "=====Convert model:" << LEVEL << " end=====" << std::endl;
    LEVEL--;
    return resulting_model;
}

std::shared_ptr<Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr& model) const {
    try {
        // std::cerr << "[   HERE   ]\n";
        auto pytorch_model = std::dynamic_pointer_cast<pytorch::InputModel>(model);
        // TODO: Remove this super-hack, tensor_map should be local for each conversion activity, see more info where
        // tensor_map is defined now
        auto model = convert_pytorch_model(pytorch_model->m_model);

        // TODO: Propose better solution for the next code block
        // Usually if nn.Module.forward is given as a source model for conversion, there is the first Parameter
        // that represents original `self` argument in forward(self, ...). `self` shouldn't play any role in model
        // inference if model is completelly frozed and all methods are inlined. So we check if it doesn't have any
        // consumers in the finally converted model and remove this parameter. This parameter should have index 0.
        if (model->get_parameters().size() > 0) {
            auto self = model->get_parameters()[0];
            if (self->output(0).get_target_inputs().empty()) {
                // There is no consumers: safe to remove
                std::cout << "[ WARNING ] Removing parameter[0] in converted Pytorch model, because it is never "
                             "used and treated as `self`\n";
                model->remove_parameter(self);
            } else {
                std::cout << "[ WARNING ] Couldn't remove parameter[0] in converted Pytorch model\n";
            }
        }

        return model;
    } catch (const std::runtime_error& e) {
        std::cerr << "[ ERROR ] Error while converting pytorch model: " << e.what() << "\n";
        std::cerr << "Rethrowing. Misleading error message from pybind11 may come next. TODO.";
        throw;
    }
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // std::cout << "[  ----- DEBUG ------ ] supported_impl with " << variants.size() << " arguments\n";
    return false;
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    // std::cout << "[  ----- DEBUG -----  ] load_impl with " << variants.size() << " parameters\n";
    if (variants.size() != 1) {
        throw std::runtime_error("Pytorch frontend supports exactly one parameter in model representation, got " +
                                 std::to_string(variants.size()) + "instead.");
    }
    auto decoder = variants[0].as<std::shared_ptr<Decoder>>();
    // std::cout << "Recognized decoder: " << decoder << "\n";
    return std::make_shared<pytorch::InputModel>(decoder);
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
