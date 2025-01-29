#include "deform_conv.hpp"
#include <openvino/frontend/onnx/frontend.hpp>
#include <openvino/opsets/opset1.hpp>

namespace ov {
namespace frontend {
namespace onnx {
namespace op {

DeformConv::DeformConv(const std::string& name, int version) 
    : ov::frontend::onnx::Op(name), m_version(version) {}

void DeformConv::operator()(const std::vector<ov::frontend::onnx::Tensor>& inputs,
                            const std::vector<ov::frontend::onnx::Attribute>& attributes,
                            std::vector<ov::frontend::onnx::Tensor>& outputs) const {
    // Validate input size, since atleast 3 are needed :)
    if (inputs.size() < 3) {
        throw std::runtime_error("DeformConv requires at least 3 inputs: data, offsets, weights.");
    }

    auto input = inputs[0];
    auto offset = inputs[1];
    auto weight = inputs[2];
    auto bias = inputs.size() > 3 ? inputs[3] : nullptr;  // Optional bias

    int64_t groups = 1;
    auto groups_attr = ov::frontend::onnx::find_attribute(attributes, "groups");
    if (groups_attr) {
        groups = groups_attr->i();
    }

    // Handling datatype differences between DeformConv-19 and DeformConv-22
    ov::element::Type output_type = input.get_element_type(); // Default to input type

    if (m_version == 22) {
        auto dtype_attr = ov::frontend::onnx::find_attribute(attributes, "dtype");
        if (dtype_attr) {
            output_type = ov::element::Type(dtype_attr->i()); // Set explicit output dtype
        }
    }

    // Create Deformable Convolution node
    auto deform_conv = std::make_shared<ov::opset1::DeformableConvolution>(
        input, offset, weight, bias,
        ov::Strides{1, 1},            
        ov::CoordinateDiff{0, 0},     
        ov::CoordinateDiff{0, 0},     
        ov::Strides{1, 1},           
        groups                         
    );

    // Set output type for DeformConv-22 if needed
    deform_conv->set_output_type(0, output_type, deform_conv->get_output_partial_shape(0));

    // Set outputs
    outputs.push_back(deform_conv);
}

}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
