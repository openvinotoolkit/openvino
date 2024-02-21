#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/matmul.hpp" 
#include "pt_framework_node.hpp"
#include "openvino/op/convert.hpp" 
#include "openvino/op/convert_like.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;

OutputVector translate_mv(const NodeContext& context) {
    num_inputs_check(context, 2, 3);  
    // "aten::mv(Tensor input, Tensor vec) -> Tensor"
    
    auto matrix = context.get_input(0);
    auto vector = context.get_input(1);

    // Perform matrix-vector multiplication
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
