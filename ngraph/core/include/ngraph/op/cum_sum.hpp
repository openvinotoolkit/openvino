// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Tensor cumulative sum operation.
            ///
            /// Compute the cumulative sum of the input tensor along the axis specified.
            ///
            /// ## Parameters
            ///
            /// |                      | Description |
            /// | -------------------- |
            /// --------------------------------------------------------------------------------------------------|
            /// | `exclusive`          | If set to 1 will return exclusive sum in which the top
            /// element
            /// is not included. 		      |
            /// |		          | In other terms, if set to 1, the j-th output element
            /// would be
            /// the
            /// sum of the first (j-1) elements.|
            /// |		          | Otherwise, it would be the sum of the first j elements.
            /// |
            ///
            /// |                      | Description                                        |
            /// | -------------------- | -------------------------------------------------- |
            /// | `reverse`            | if set to 1, performs the sum in reverse direction |
            ///
            /// ## Inputs
            ///
            /// |       | Description                                            |
            /// | ----- | ------------------------------------------------------ |
            /// | `arg` | An input tensor of any shape and numeric element type. |
            ///
            /// |       | Description |
            /// | ----- |
            /// ------------------------------------------------------------------------------------------------|
            /// | `axis`| zero dimension tensor specifying axis position along which cumulative sum
            /// must
            /// be performed.    |
            ///
            /// ## Output
            ///
            /// | Description |
            /// |
            /// ------------------------------------------------------------------------------------|
            /// | Output tensor of the same type as `arg` with cumulative sums of the arg's elements
            /// |

            class NGRAPH_API CumSum : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"CumSum", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a cumulative summation operation.
                CumSum() = default;

                /// \brief Constructs a cumulative summation operation.
                ///
                /// \param arg The tensor to be summed.
                /// \param axis zero dimension tensor specifying axis position along which
                /// cumulative sum must be performed
                /// \param exclusive if set to true, the top element is not included
                /// \param reverse if set to true, will perform the sums in reverse direction
                CumSum(const Output<Node>& arg,
                       const Output<Node>& axis,
                       const bool exclusive = false,
                       const bool reverse = false);

                /// \brief Constructs a cumulative summation operation with axis = 0
                ///
                /// \param arg The tensor to be summed
                CumSum(const Output<Node>& arg,
                       const bool exclusive = false,
                       const bool reverse = false);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                /// \return The default value for CumSum.
                virtual std::shared_ptr<Node> get_default_value() const override;
                bool is_exclusive() const { return m_exclusive; }
                bool is_reverse() const { return m_reverse; }

            private:
                bool m_exclusive;
                bool m_reverse;
            };
        } // namespace v0
        using v0::CumSum;
    } // namespace op
} // namespace ngraph
