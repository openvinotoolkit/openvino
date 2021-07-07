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
        namespace v7
        {
            /// \brief Tensor roll operation.
            class NGRAPH_API Roll : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                Roll() = default;

                ///
                /// \brief      Constructs a roll operation.
                ///
                /// \param      data         Node producing the tensor to be shifted.
                /// \param      shift        Node producing the 0D or 1D tensor which specifies the
                /// number of places by which the elements are shifted.
                /// \param      axes         Node producing the 0D or 1D tensor which specifies axes
                /// along which elements are shifted.
                ///
                Roll(const Output<Node>& data, const Output<Node>& shift, const Output<Node>& axes);

                void validate_and_infer_types() override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v7
    }     // namespace op
} // namespace ngraph
