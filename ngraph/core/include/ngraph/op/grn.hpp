// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief  Global Response Normalization with L2 norm (across channels only).
            ///
            class NGRAPH_API GRN : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                GRN() = default;
                /// \brief      Constructs a GRN operation.
                ///
                /// \param      data  - Node producing the input tensor
                /// \param      bias  - The bias added to the variance.
                ///
                GRN(const Output<Node>& data, float bias);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                float get_bias() const { return m_bias; }

            protected:
                float m_bias = 1.0f;
            };
        } // namespace v0
        using v0::GRN;
    } // namespace op
} // namespace ngraph
