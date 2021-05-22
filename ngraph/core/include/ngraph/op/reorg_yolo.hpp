// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API ReorgYolo : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReorgYolo", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ReorgYolo() = default;
                /// \brief Constructs a ReorgYolo operation
                ///
                /// \param input          Input
                /// \param stride         Stride to reorganize input by
                ReorgYolo(const Output<Node>& input, const size_t stride);

                // Constructor with `strides` for backward compatibility
                ReorgYolo(const Output<Node>& input, const Strides& strides);

                void validate_and_infer_types() override;

                virtual bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                Strides get_strides() const { return m_strides; }

            private:
                Strides m_strides;
            };
        } // namespace v0
        using v0::ReorgYolo;
    } // namespace op
} // namespace ngraph
