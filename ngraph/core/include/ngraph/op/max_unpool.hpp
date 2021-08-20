
#pragma once

#include "ngraph/op/op.hpp"
namespace ngraph
{
    namespace op
    {
        namespace v8
        {
            class NGRAPH_API MaxPoolGrad : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                MaxPoolGrad() = default;

                MaxPoolGrad(const ngraph::Output<ngraph::Node>& poolInp,
                   const ngraph::Output<ngraph::Node>& poolOut,
                   const ngraph::Output<ngraph::Node>& inp,
                   const ngraph::Output<ngraph::Node>& shape);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

                bool has_evaluate() const override;
            };

        } // namespace v8
    }     // namespace op
} // namespace ngraph
