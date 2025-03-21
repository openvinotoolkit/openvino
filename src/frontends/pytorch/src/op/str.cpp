#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "pt_framework_node.hpp"
#include "openvino/core/validation_util.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_str(const NodeContext& context) {
    // Get the input tensor
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    // Check if the input tensor is of a supported type
    if (input.get_element_type() != ov::element::string &&
        input.get_element_type() != ov::element::f32 &&
        input.get_element_type() != ov::element::i32) {
        OPENVINO_THROW("Unsupported tensor type for str operation");
    }
    // if the tensor is constant
    //  Try to get the constant value of the input tensor
    auto constant_value = util::get_constant_from_source(input);
    if (constant_value) {
        // Convert the constant value to a string
        std::string str_value;
        if (input.get_element_type() == ov::element::f32) {
            str_value = std::to_string(constant_value->cast_vector<float>()[0]);
        } else if (input.get_element_type() == ov::element::i32) {
            str_value = std::to_string(constant_value->cast_vector<int32_t>()[0]);
        } else if (input.get_element_type() == ov::element::string) {
            str_value = constant_value->cast_vector<std::string>()[0];
        }
        // Create a new Constant node with the string value
        auto str_node = std::make_shared<ov::op::v0::Constant>(
            ov::element::string,
            ov::Shape{1},
            std::vector<std::string>{str_value});
        return {context.mark_node(str_node)};
    }
    // if the tensor is not constant
    auto shape = context.mark_node(std::make_shared<ov::op::v3::ShapeOf>(input, element::i64));
    auto dtype = input.get_element_type();
    // OpenVINO is able to handle numerical data, and does not support string data natively
    // therefore, we capture tensor's shape and datatype instead which can be later used in debugging
    // Utilising the PtFrameworkNode for this purpose

    // from the documentation
    // https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/pytorch/README.md
    // PtFrameworkNode is used to represent unconverted operation from the original model.
    ov::op::util::FrameworkNodeAttrs attrs;
    attrs[PtFrameworkNode::op_type_key] = "aten::str";
    attrs["dtype"] = dtype.get_type_name();
    
    auto decoder = context.get_decoder();
    auto ptf_node = std::make_shared<PtFrameworkNode>(decoder, OutputVector{input}, 1);
    ptf_node->set_attrs(attrs);

    return {context.mark_node(ptf_node)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov