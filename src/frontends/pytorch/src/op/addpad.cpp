#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"
#include <iostream>

using namespace InferenceEngine;
namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
int main() {
    try {
        // Padding values
        std::vector<float> padding_values_1d = {1.0}; // Padding for 1D tensor
        std::vector<float> padding_values_2d = {1.0, 2.0, 3.0, 4.0}; // Padding for 2D tensor
        std::vector<float> padding_values_3d = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}; // Padding for 3D tensor

        // Input tensors
        auto input_tensor_1d = op::Constant::create(element::f32, Shape{5}, {1.0, 2.0, 3.0, 4.0, 5.0}); // 1D tensor
        auto input_tensor_2d = op::Constant::create(element::f32, Shape{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}); // 2D tensor
        auto input_tensor_3d = op::Constant::create(element::f32, Shape{2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}); // 3D tensor

        // Create padding tensors
        auto padding_tensor_1d = op::Constant::create(element::f32, Shape{1}, padding_values_1d);
        auto padding_tensor_2d = op::Constant::create(element::f32, Shape{2, 2}, padding_values_2d);
        auto padding_tensor_3d = op::Constant::create(element::f32, Shape{3, 3}, padding_values_3d);

        // Perform addition with padding for 1D tensor
        auto add_op_1d = std::make_shared<op::v1::Add>(input_tensor_1d, padding_tensor_1d);
        auto result_tensor_1d = add_op_1d->evaluate({});

        // Perform addition with padding for 2D tensor
        auto add_op_2d = std::make_shared<op::v1::Add>(input_tensor_2d, padding_tensor_2d);
        auto result_tensor_2d = add_op_2d->evaluate({});

        // Perform addition with padding for 3D tensor
        auto add_op_3d = std::make_shared<op::v1::Add>(input_tensor_3d, padding_tensor_3d);
        auto result_tensor_3d = add_op_3d->evaluate({});

        // Display results
        std::cout << "Result after adding padding for 1D tensor:" << std::endl;
        std::cout << result_tensor_1d->get_shape() << ": " << result_tensor_1d->cast_vector<float>() << std::endl;

        std::cout << "Result after adding padding for 2D tensor:" << std::endl;
        std::cout << result_tensor_2d->get_shape() << ": " << result_tensor_2d->cast_vector<float>() << std::endl;

        std::cout << "Result after adding padding for 3D tensor:" << std::endl;
        std::cout << result_tensor_3d->get_shape() << ": " << result_tensor_3d->cast_vector<float>() << std::endl;

        std::cout << "Inference completed successfully." << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov