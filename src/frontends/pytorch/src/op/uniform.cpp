#include <openvino/frontend/pytorch/node_context.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/random_uniform.hpp>
#include <openvino/op/shape_of.hpp>

#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_uniform(const NodeContext& context) {
    // Ensure correct number of inputs (self, from, to are optional)
    num_inputs_check(context, 1, 3);
    auto input = context.get_input(0);

    // Get low (from) and high (to) values with defaults
    float from = 0.0f;
    float to = 1.0f;
    if (context.get_input_size() > 1) {
        from = context.const_input<float>(1);
    }
    if (context.get_input_size() > 2) {
        to = context.const_input<float>(2);
    }

    // Get input tensor's shape and dtype
    auto shape = context.mark_node(std::make_shared<ov::op::v3::ShapeOf>(input));
    auto dtype = input.get_element_type();

    // Create min and max constants with the same dtype as input
    auto min_val = context.mark_node(ov::op::v0::Constant::create(dtype, ov::Shape{}, {from}));
    auto max_val = context.mark_node(ov::op::v0::Constant::create(dtype, ov::Shape{}, {to}));

    // Create RandomUniform node
    auto random_uniform =
        context.mark_node(std::make_shared<ov::op::v8::RandomUniform>(shape, min_val, max_val, dtype, 0, 0));

    return {random_uniform};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
