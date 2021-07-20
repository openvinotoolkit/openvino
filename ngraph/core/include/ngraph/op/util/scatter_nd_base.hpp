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
            ///
            /// \brief      Base class for ScatterNDXXX operators.
            ///
            class NGRAPH_API ScatterNDBase : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScatterNDBase", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                // Respective input ordinal number.
                static constexpr int INPUTS = 0;
                static constexpr int INDICES = 1;
                static constexpr int UPDATES = 2;
                virtual void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

            protected:
                ScatterNDBase() = default;

                ///
                /// \brief      Constructs ScatterNDBase object.
                ///
                /// \param      inputs   The input tensor to be updated.
                /// \param      indices  The tensor with indexes which will be updated.
                /// \param      updates  The tensor with update values.
                ///
                ScatterNDBase(const Output<Node>& inputs,
                              const Output<Node>& indices,
                              const Output<Node>& updates);
            };
        } // namespace util
    }     // namespace op
} // namespace ngraph
