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
            /// \brief      Base class for ScatterXXX operators.
            ///
            class NGRAPH_API ScatterBase : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScatterBase", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                virtual void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

            protected:
                ScatterBase() = default;

                ///
                /// \brief      Constructs ScatterBase object.
                ///
                /// \param      inputs   The input tensor to be updated.
                /// \param      indices  The tensor with indexes which will be updated.
                /// \param      updates  The tensor with update values.
                /// \param[in]  axis     The axis at which elements will be updated.
                ///
                ScatterBase(const Output<Node>& inputs,
                            const Output<Node>& indices,
                            const Output<Node>& updates,
                            const Output<Node>& axis);

            private:
                // Respective input ordinal number.
                static constexpr int DATA = 0;
                static constexpr int INDICES = 1;
                static constexpr int UPDATES = 2;
                static constexpr int AXIS = 3;
            };
        } // namespace util
    }     // namespace op
} // namespace ngraph
