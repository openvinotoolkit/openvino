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
            /// \brief Adaptive max pooling operation.
            ///
            class NGRAPH_API AdaptiveMaxPool : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                AdaptiveMaxPool() = default;

                ///
                /// \brief    Constructs adaptive max pooling operation.
                ///
                /// \param    data                  Input data
                ///
                /// \param    output_shape          1D tensor describing output shape for spatial
                ///                                 dimensions.
                ///
                /// \param    index_element_type    Specifies the output tensor type for indices
                /// output
                ///
                AdaptiveMaxPool(
                    const Output<Node>& data,
                    const Output<Node>& output_shape,
                    const ngraph::element::Type& index_element_type = ngraph::element::i64);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                element::Type get_index_element_type() const { return m_index_element_type; }

            protected:
                ngraph::element::Type m_index_element_type = ngraph::element::i64;
            };
        } // namespace v8
    }     // namespace op
} // namespace ngraph
