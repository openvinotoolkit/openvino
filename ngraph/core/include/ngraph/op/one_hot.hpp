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
            class NGRAPH_API OneHot : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"OneHot", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a one-hot operation.
                OneHot() = default;
                /// \brief Constructs a one-hot operation.
                ///
                /// \param indices   Input tensor containing indices.
                /// \param depth     Specifies number of classes and the size of one-hot dimension.
                /// \param on_value  Specifies value that the locations in output tensor represented
                ///                  by indices in input take.
                /// \param off_value Specifies value that the locations in output tensor not
                /// represented
                ///                  by indices in input take.
                /// \param axis      Axis along which one-hot representation in added.
                OneHot(const Output<Node>& indices,
                       const Output<Node>& depth,
                       const Output<Node>& on_value,
                       const Output<Node>& off_value,
                       int64_t axis);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                void validate_and_infer_types() override;

                virtual bool evaluate(const HostTensorVector& output_values,
                                      const HostTensorVector& input_values) const override;
                bool has_evaluate() const override;

                /// \return The index of the one-hot axis.
                int64_t get_axis() const { return m_axis; }
                void set_axis(int64_t axis) { m_axis = axis; }

            protected:
                int64_t m_axis;
            };
        } // namespace v1
    }     // namespace op
} // namespace ngraph
