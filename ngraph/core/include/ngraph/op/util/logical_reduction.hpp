// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            /// \brief Abstract base class for logical reduction operations, i.e., operations where
            ///        chosen axes of the input tensors are eliminated (reduced out) by repeated
            ///        application of a particular binary logical operation.
            class NGRAPH_API LogicalReduction : public Op
            {
            protected:
                /// \brief Constructs a logical reduction operation.
                LogicalReduction();
                /// \brief Constructs a logical reduction operation.
                ///
                /// \param arg Output that produces the first input tensor.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                LogicalReduction(const Output<Node>& arg, const AxisSet& reduction_axes);
                /// \brief Constructs a 'dynamic' logical reduction operation.
                ///
                /// \param arg Node that produces the first input tensor.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                LogicalReduction(const Output<Node>& arg, const Output<Node>& reduction_axes);

            public:
                NGRAPH_RTTI_DECLARATION;
                void validate_and_infer_types() override;

                /// \return true if reduction axes are constant else false.
                bool reduction_axes_constant() const;

                /// \return The axis positions (0-based) to be eliminated through reduction.
                /// \throws CheckFailure if the reduction axes are not constant. (Use
                ///           reduction_axes_constant to check.)
                const AxisSet get_reduction_axes() const;
                void set_reduction_axes(const AxisSet& reduction_axes);
            };
        } // namespace util
    }     // namespace op
} // namespace ngraph
