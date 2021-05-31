// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Product reduction operation.
            ///
            /// Reduces the tensor, eliminating the specified reduction axes by taking the product.
            class NGRAPH_API ReduceProd : public util::ArithmeticReductionKeepDims
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                /// \brief Constructs a product reduction operation.
                ReduceProd() = default;
                /// \brief Constructs a product reduction operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                /// \param keep_dims If set to true it holds axes that are used for reduction.
                ReduceProd(const Output<Node>& arg,
                           const Output<Node>& reduction_axes,
                           bool keep_dims = false);

                size_t get_version() const override { return 1; }
                /// \return The default value for Product.
                virtual std::shared_ptr<Node> get_default_value() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
                bool evaluate_lower(const HostTensorVector& outputs) const override;
                bool evaluate_upper(const HostTensorVector& outputs) const override;
            };
        } // namespace v1
    }     // namespace op
} // namespace ngraph
