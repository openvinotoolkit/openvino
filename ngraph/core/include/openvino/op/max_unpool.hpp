
#pragma once

#include "openvino/op/op.hpp"
namespace ov {
namespace op {
namespace v8 {
class OPENVINO_API MaxUnpool : public Op {
public:
    OPENVINO_OP("MaxUnpool", "opset8");
    BWDCMP_RTTI_DECLARATION;

    MaxUnpool() = default;

    MaxUnpool(const ngraph::Output<ngraph::Node>& poolInp,
              const ngraph::Output<ngraph::Node>& poolOut,
              const ngraph::Output<ngraph::Node>& inp,
              const ngraph::Output<ngraph::Node>& shape,
              const Strides& strides,
              const Shape& pads_begin,
              const Shape& pads_end,
              const Shape& kernel);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    virtual std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;

    bool has_evaluate() const override;

    Shape m_kernel, m_pads_begin, m_pads_end, m_strides;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
