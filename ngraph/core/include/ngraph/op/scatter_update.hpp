// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/scatter_base.hpp"
#include "ngraph/runtime/host_tensor.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v3
        {
            ///
            /// \brief      Set new values to slices from data addressed by indices
            ///
            class NGRAPH_API ScatterUpdate : public util::ScatterBase
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScatterUpdate", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ScatterUpdate() = default;
                ///
                /// \brief      Constructs ScatterUpdate operator object.
                ///
                /// \param      data     The input tensor to be updated.
                /// \param      indices  The tensor with indexes which will be updated.
                /// \param      updates  The tensor with update values.
                /// \param[in]  axis     The axis at which elements will be updated.
                ///
                ScatterUpdate(const Output<Node>& data,
                              const Output<Node>& indices,
                              const Output<Node>& updates,
                              const Output<Node>& axis);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& inputs) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            private:
                bool evaluate_scatter_update(const HostTensorVector& outputs,
                                             const HostTensorVector& inputs) const;
            };
        } // namespace v3
    }     // namespace op
} // namespace ngraph
