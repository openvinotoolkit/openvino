// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v4
        {
            /// \brief Reduction operation using L2 norm:
            ///
            /// Reduces the tensor, eliminating the specified reduction axes by taking the L2-norm.
            class NGRAPH_API ReduceL2 : public util::ArithmeticReductionKeepDims
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReduceL2", 4};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a reducet L2-norm operation.
                ReduceL2() = default;
                /// \brief Constructs a reduce L2-norm operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                /// \param keep_dims If set to true it holds axes that are used for reduction.
                ReduceL2(const Output<Node>& arg,
                         const Output<Node>& reduction_axes,
                         bool keep_dims = false);

                size_t get_version() const override { return 4; }
                /// \return The default value for Reduce.
                virtual std::shared_ptr<Node> get_default_value() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
            };
        }
    }
}
