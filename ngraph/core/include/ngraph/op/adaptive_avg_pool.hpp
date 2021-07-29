// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v8
        {
            /// \brief Adaptive average pooling operation.
            ///
            class NGRAPH_API AdaptiveAvgPool : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                AdaptiveAvgPool() = default;

                ///
                /// \brief    Constructs adaptive average pooling operation.
                ///
                /// \param    data            Input data
                ///
                /// \param    output_shape    1D tensor describing output shape for spatial
                ///                           dimensions.
                ///
                AdaptiveAvgPool(const Output<Node>& data, const Output<Node>& output_shape);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v8
    }     // namespace op
} // namespace ngraph
