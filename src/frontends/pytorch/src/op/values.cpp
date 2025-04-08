#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector aten_values(const NodeContext& context) {
    num_inputs_check(context, 1, 1);  

    auto input_tensor = context.get_input(0);

    if (input_tensor.get_element_type().is_real()) {
        return {context.mark_node(input_tensor)};
    }

    auto values = context.mark_node(std::make_shared<v0::Gather>(
        input_tensor,
        v0::Constant::create(element::i32, Shape{1}, {1}),  
        v0::Constant::create(element::i32, Shape{}, {0})    
    ));

    return {values};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

