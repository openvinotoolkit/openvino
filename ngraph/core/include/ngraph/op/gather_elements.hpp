// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v6
        {
            /// \brief GatherElements operation
            ///
            class NGRAPH_API GatherElements : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                GatherElements() = default;

                /// \brief Constructs a GatherElements operation.
                ///
                /// \param data Node producing data that are gathered
                /// \param indices Node producing indices by which the operation gathers elements
                /// \param axis specifies axis along which indices are specified
                GatherElements(const Output<Node>& data,
                               const Output<Node>& indices,
                               const int64_t axis);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                int64_t get_axis() const { return m_axis; }

            private:
                int64_t m_axis;
            };
        } // namespace v6
    }     // namespace op
} // namespace ngraph
