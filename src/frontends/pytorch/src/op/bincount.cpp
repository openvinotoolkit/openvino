#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/maximum.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_bincount(const NodeContext& context) {
    num_inputs_check(context, 3, 3);

    auto input = context.get_input(0); 
    auto weights = context.get_input(1); 
    auto minlength = context.get_input(2); 
    
    auto input_int = context.mark_node(std::make_shared<v0::Convert>(input, element::i32));

    auto max_val = context.mark_node(std::make_shared<v1::ReduceMax>(input_int, ov::AxisSet{0}, true));
    auto max_length = context.mark_node(std::make_shared<v1::Add>(max_val, context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}))));
    auto output_size = context.mark_node(std::make_shared<v1::Maximum>(max_length, minlength));

    auto output = context.mark_node(v0::Constant::create(element::f32, Shape{output_size}, {0}));

    auto weight_tensor = weights.get_node_shared_ptr() != nullptr
                         ? weights
                         : context.mark_node(v0::Constant::create(element::f32, Shape{input->get_shape()}, {1}));

    auto indices = context.mark_node(std::make_shared<v0::Unsqueeze>(input_int, v0::Constant::create(element::i64, Shape{1}, {1})));

    auto scatter_result = context.mark_node(std::make_shared<v3::ScatterNDUpdate>(output, indices, weight_tensor));

    return {scatter_result};
}

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov
