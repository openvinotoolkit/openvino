#include "openvino/op/relu.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/quantized_linear_relu.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/add.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

ov::OutputVector translate_quantized_relu(const NodeContext& context) {
    auto input = context.get_input(0);  // Quantized input tensor

    // Step 1: Dequantize input tensor (convert from int8/uint8 to float32)
    auto dequantized_input = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);

    // Step 2: Apply ReLU activation (ReLU(x) = max(0, x))
    auto relu_output = std::make_shared<ov::op::v0::Relu>(dequantized_input);

    // Step 3: Requantize the output tensor (convert back to original quantized type)
    auto requantized_output = std::make_shared<ov::op::v0::Convert>(relu_output, input.get_element_type());

    return {requantized_output};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
