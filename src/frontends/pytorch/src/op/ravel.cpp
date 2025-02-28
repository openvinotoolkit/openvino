#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_ravel(const NodeContext& context) {
    // Ensure exactly one input is provided
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    
    // Create shape tensor [-1] for flattening
    auto shape = ov::op::v0::Constant::create(element::i64, Shape{1}, {-1});
    
    // Reshape input tensor to a 1D tensor
    return {std::make_shared<ov::op::v1::Reshape>(input, shape, false)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
