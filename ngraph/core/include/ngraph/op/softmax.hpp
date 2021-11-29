// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            class NGRAPH_API Softmax : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Softmax", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Softmax()
                    : m_axis(0)
                {
                }
                /// \brief Constructs a softmax operation.
                ///
                /// \param arg Node that produces the first input tensor.<br>
                /// `[d0, ...]`
                /// \param axis The axis position (0-based) on which to calculate the softmax.
                ///
                /// Output `[d0, ...]`
                ///
                Softmax(const Output<Node>& arg, const size_t axis = 1);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                size_t get_version() const override { return 1; }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_axis() const { return m_axis; }
                void set_axis(const size_t axis) { m_axis = axis; }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            private:
                size_t m_axis;
            };
        } // namespace v1
    }     // namespace op
} // namespace ngraph
