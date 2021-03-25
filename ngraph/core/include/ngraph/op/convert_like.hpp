// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise type conversion operation.
            class NGRAPH_API ConvertLike : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ConvertLike", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a conversion operation.
                ConvertLike() = default;
                /// \brief Constructs a conversion operation.
                /// \param data  Node that produces the input tensor.
                /// \param like  Node which provides the target type information for the conversion.
                ConvertLike(const Output<Node>& data, const Output<Node>& like);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& input_values) override;
            };

        } // namespace v1

    } // namespace op

} // namespace ngraph
