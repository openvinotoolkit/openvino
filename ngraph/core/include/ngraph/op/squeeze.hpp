// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API Squeeze : public ngraph::op::util::FusedOp
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                Squeeze();
                Squeeze(const Output<Node>& data, const Output<Node>& axes);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual OutputVector decompose_op() const override;
                virtual void pre_validate_and_infer_types() override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool evaluate_lower(const HostTensorVector& outputs) const override;
                bool evaluate_upper(const HostTensorVector& outputs) const override;
                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& inputs_values) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool is_dynamic() const override;
            };
        }
        using v0::Squeeze;
    }
}

NGRAPH_SUPPRESS_DEPRECATED_END
