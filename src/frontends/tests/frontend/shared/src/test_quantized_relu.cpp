#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/quantized_linear_relu.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

class QuantizedLinearReLUTest : public ::testing::Test {
protected:
    // Helper function to create a quantized tensor with some values
    std::shared_ptr<ov::op::v0::Constant> create_quantized_tensor() {
        // Example: Create a constant tensor with values that will test the relu functionality
        std::vector<int8_t> data = {-5, -2, 0, 2, 5};  // Example quantized values
        auto tensor = ov::Tensor(ov::element::i8, ov::Shape{5}, data);
        return std::make_shared<ov::op::v0::Constant>(tensor);
    }

    // Helper function to check if the output matches expected quantized values
    void check_quantized_output(const ov::Output<ov::Node>& output, const std::vector<int8_t>& expected_values) {
        auto output_tensor = output.get_node()->output(0).get_tensor();
        auto output_data = output_tensor.data<int8_t>();

        for (size_t i = 0; i < expected_values.size(); ++i) {
            EXPECT_EQ(output_data[i], expected_values[i]) << "Mismatch at index " << i;
        }
    }
};

TEST_F(QuantizedLinearReLUTest, QuantizedLinearReLU) {
    // Step 1: Create a quantized input tensor
    auto input = create_quantized_tensor();
    
    // Step 2: Apply quantized linear ReLU using the function we implemented
    ov::OutputVector result = translate_quantized_relu(NodeContext(input));

    // Step 3: Define the expected output for this case (after ReLU and requantization)
    std::vector<int8_t> expected_output = {0, 0, 0, 2, 5};  // Expected result after ReLU

    // Step 4: Check the output
    check_quantized_output(result[0], expected_output);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

