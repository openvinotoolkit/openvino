#ifndef OPENVINO_FRONTENDS_ONNX_OP_DEFORM_CONV_HPP
#define OPENVINO_FRONTENDS_ONNX_OP_DEFORM_CONV_HPP

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {

class DeformConv : public Op {
public:
    OPENVINO_OP("DeformConv", "opset1", op::Op);
    DeformConv() = default;
    DeformConv(const Output<Node>& data, const Output<Node>& offsets, const Output<Node>& weights);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Output<Node>& get_data() const { return input_value(0); }
    const Output<Node>& get_offsets() const { return input_value(1); }
    const Output<Node>& get_weights() const { return input_value(2); }
};

}  // namespace v1
}  // namespace op
}  // namespace ov

#endif  // OPENVINO_FRONTENDS_ONNX_OP_DEFORM_CONV_HPP
