// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/logical_reduction_keep_dims.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Performs a reduction using "logical or"
            ///
            /// The reduction is performed over slices of the first input. The slices shape depends
            /// on the values passed to the second input - the axes.
            class NGRAPH_API ReduceLogicalOr : public util::LogicalReductionKeepDims
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                ReduceLogicalOr() = default;
                /// \brief Constructs a ReduceLogicalOr node.
                ///
                /// \param data - The input tensor with data to be reduced
                /// \param reduction_axes - The input tensor with information about axes over which
                /// the first tensor should be sliced prior to the reduction operation
                /// \param keep_dims - Indicates if the axes used for reduction should be held/kept
                ReduceLogicalOr(const Output<Node>& data,
                                const Output<Node>& reduction_axes,
                                const bool keep_dims = false);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v1
    }     // namespace op
} // namespace ngraph
