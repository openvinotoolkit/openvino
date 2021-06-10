// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v4
        {
            /// \brief Range operation, analogous to `arange()` in Numpy.
            class NGRAPH_API Range : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                /// \brief Constructs an unitialized range operation.
                Range() = default;

                /// \brief Constructs a range operation.
                ///
                /// \param start The tensor producing the start value. Must be a scalar of numeric
                ///              element type.
                /// \param stop The tensor producing the stop value. Must be a scalar of numeric
                ///             element type.
                /// \param step The tensor producing the step value. Must be a scalar of numeric
                ///             element type.
                /// \param output_type The type of the output.
                Range(const Output<Node>& start,
                      const Output<Node>& stop,
                      const Output<Node>& step,
                      element::Type output_type);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            private:
                element::Type m_output_type;
            };
        } // namespace v4
        namespace v0
        {
            /// \brief Range operation, analogous to `range()` in Python.
            class NGRAPH_API Range : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Range", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs an unitialized range operation.
                Range() = default;

                /// \brief Constructs a range operation.
                ///
                /// \param start The tensor producing the start value. Must be a scalar of integer
                ///              element type, and same element type as `stop` and `step`.
                /// \param stop The tensor producing the stop value. Must be a scalar of integer
                ///             element type, and same element type as `start` and `step`.
                /// \param step The tensor producing the step value. Must be a scalar of integer
                ///             element type, and same element type as `start` and `stop`.
                Range(const Output<Node>& start,
                      const Output<Node>& stop,
                      const Output<Node>& step);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v0
        using v0::Range;
    } // namespace op
} // namespace ngraph
