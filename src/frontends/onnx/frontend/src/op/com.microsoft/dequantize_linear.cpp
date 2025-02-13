
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

ov::OutputVector dequantizelinear(const ov::frontend::onnx::Node& node) {
    // Documentation :
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftdequantizelinear

    common::default_op_checks(node, 2);

    const auto inputs = node.get_ov_inputs();
    auto x = inputs[0];
    auto x_scale = inputs[1];

    auto x_zero_point = (inputs.size() > 2) ? inputs[2] : v0::Constant::create(x.get_element_type(), {}, {0});

    const auto axis = node.get_attribute_value<int64_t>("axis", -1);

    if (axis == -1) {
        CHECK_VALID_NODE(node,
                         x_scale.get_node()->get_shape().size() == 0,
                         "x_scale must be a scalar for per-tensor quantization.");
        CHECK_VALID_NODE(node,
                         x_zero_point.get_shape().size() == 0,
                         "x_zero_point must be a scalar for per-tensor quantization.");
    } else {
        CHECK_VALID_NODE(node,
                         x_scale.get_node()->get_shape().size() == 1,
                         "x_scale must be a 1-D tensor for per-axis quantization.");
        CHECK_VALID_NODE(node,
                         x_zero_point.get_shape().size() == 1,
                         "x_zero_point must be a 1-D tensor for per-axis quantization.");
        CHECK_VALID_NODE(node,
                         x_scale.get_node()->get_shape() == x_zero_point.get_shape(),
                         "x_scale and x_zero_point must have the same shape.");
        auto shape_of_x = std::make_shared<v3::ShapeOf>(x, element::i64);
        auto axis_dim = std::make_shared<v8::Gather>(shape_of_x,
                                                     v0::Constant::create(element::i64, Shape{1}, {axis}),
                                                     v0::Constant::create(element::i64, Shape{}, {0}));
        auto reshaped_scale =
            std::make_shared<v1::Reshape>(x_scale, v0::Constant::create(element::i64, Shape{1}, {-1}), true);

        auto reshaped_zero_point =
            std::make_shared<v1::Reshape>(x_zero_point, v0::Constant::create(element::i64, Shape{1}, {-1}), true);
        x_scale = std::make_shared<v3::Broadcast>(reshaped_scale, shape_of_x);
        x_zero_point = std::make_shared<v3::Broadcast>(reshaped_zero_point, shape_of_x);
    }

    auto dequantized_data = std::make_shared<v1::Subtract>(x, x_zero_point);
    auto dequantize_data_float = std::make_shared<v0::Convert>(dequantized_data, x_scale.get_element_type());
    auto result = std::make_shared<v1::Multiply>(dequantize_data_float, x_scale);

    return {result};
}

ONNX_OP("DequantizeLinear", OPSET_SINCE(1), com_microsoft::opset_1::dequantizelinear, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
