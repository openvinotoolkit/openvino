// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            class NGRAPH_API ReduceMean : public util::ArithmeticReductionKeepDims
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                ReduceMean() = default;

                /// \param arg The tensor to be summed.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                /// \param keep_dims If set to 1 it holds axes that are used for reduction.
                ReduceMean(const Output<Node>& arg,
                           const Output<Node>& reduction_axes,
                           bool keep_dims = false);

                size_t get_version() const override { return 1; }
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v1
    }     // namespace op
} // namespace ngraph
