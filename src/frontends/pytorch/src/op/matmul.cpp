 #include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"


namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;
OutputVector translate_mv(const NodeContext& context) {
    num_inputs_check(context, 3, 7);  
    auto matrix = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), element::f32));
    auto vector = context.mark_node(std::make_shared<v0::Convert>(context.get_input(1), element::f32));
    auto result = context.mark_node(std::make_shared<v1::Multiply>(matrix, vector));  // Custom operation: element-wise multiplication
    return {result};
}
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov