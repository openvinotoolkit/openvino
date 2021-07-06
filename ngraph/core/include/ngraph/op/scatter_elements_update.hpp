// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/host_tensor.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v3
        {
            class NGRAPH_API ScatterElementsUpdate : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScatterElementsUpdate", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ScatterElementsUpdate() = default;
                /// \brief Constructs a ScatterElementsUpdate node

                /// \param data            Input data
                /// \param indices         Data entry index that will be updated
                /// \param updates         Update values
                /// \param axis            Axis to scatter on
                ScatterElementsUpdate(const Output<Node>& data,
                                      const Output<Node>& indices,
                                      const Output<Node>& updates,
                                      const Output<Node>& axis);

                virtual void validate_and_infer_types() override;
                virtual bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& inputs) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            private:
                bool evaluate_scatter_element_update(const HostTensorVector& outputs,
                                                     const HostTensorVector& inputs) const;
            };
        } // namespace v3
        using v3::ScatterElementsUpdate;
    } // namespace op
} // namespace ngraph
