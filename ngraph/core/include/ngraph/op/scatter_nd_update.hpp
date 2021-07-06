// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/scatter_nd_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v3
        {
            /// \brief Add updates to slices from inputs addressed by indices
            class NGRAPH_API ScatterNDUpdate : public util::ScatterNDBase
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScatterNDUpdate", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ScatterNDUpdate() = default;
                /// \param inputs Tensor
                /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
                /// \param updates Tensor: Must have same type as inputs
                ScatterNDUpdate(const Output<Node>& inputs,
                                const Output<Node>& indices,
                                const Output<Node>& updates)
                    : util::ScatterNDBase(inputs, indices, updates)
                {
                }

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v3
        using v3::ScatterNDUpdate;
    } // namespace op
} // namespace ngraph
