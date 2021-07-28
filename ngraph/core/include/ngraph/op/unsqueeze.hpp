// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API Unsqueeze : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                Unsqueeze() = default;
                Unsqueeze(const Output<Node>& data, const Output<Node>& axes);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
                bool evaluate_lower(const HostTensorVector& output_values) const override;
                bool evaluate_upper(const HostTensorVector& output_values) const override;

                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& inputs_values) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v0
        using v0::Unsqueeze;
    } // namespace op
} // namespace ngraph
