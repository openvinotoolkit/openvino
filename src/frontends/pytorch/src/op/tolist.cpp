#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_tolist(const NodeContext& context){
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    auto input_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(input.get_node_shared_ptr());
    if (!input_const) {
        FRONT_END_OP_CONVERSION_CHECK(false, "prim::tolist: only constant tensors are supported currently.");
    }

    std::vector<Output<Node>> elements;
    if (input_const->get_element_type() == element::i32) {
        auto values = input_const->cast_vector<int32_t>();
        for (int32_t v : values) {
            elements.push_back(v0::Constant::create(element::i32, Shape{}, {v}));
        }
    } 
    else if (input_const->get_element_type() == element::f32) {
        auto values = input_const->cast_vector<float>();
        for (float v : values) {
            elements.push_back(v0::Constant::create(element::f32, Shape{}, {v}));
        }
    } 
    else {
        FRONT_END_OP_CONVERSION_CHECK(false, "prim::tolist: unsupported element type.");
    }

    // Pack as tuple
    auto tuple = std::make_shared<ov::op::v0::Tuple>(elements);
    return {context.mark_node(tuple)};
    
}

}
}
}
}