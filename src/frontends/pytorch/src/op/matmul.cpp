#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/matmul.hpp" 
#include "pt_framework_node.hpp"
#include "openvino/op/convert.hpp" 
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;

OutputVector translate_mv(const NodeContext& context) {
    num_inputs_check(context, 2, 3);  

    auto matrix = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), element::f32));
    auto vector = context.mark_node(std::make_shared<v0::Convert>(context.get_input(1), element::f32));


    auto result = context.mark_node(std::make_shared<v1::MatMul>(matrix, vector));



    if (!context.input_is_none(2)) {
        context.mutate_input(2, result);
    }

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

}  // namespace ov
