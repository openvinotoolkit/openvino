// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v5
        {
            /// \brief GatherND operation
            ///
            class NGRAPH_API GatherND : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                GatherND() = default;

                /// \brief Constructs a GatherND operation.
                ///
                /// \param data Node producing data that are gathered
                /// \param indices Node producing indices by which the operation gathers elements
                /// or slices from data
                /// \param batch_dims Specifies a number of batch dimensions
                GatherND(const Output<Node>& data,
                         const Output<Node>& indices,
                         const size_t batch_dims = 0);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_batch_dims() const { return m_batch_dims; }

            private:
                size_t m_batch_dims;
            };
        } // namespace v5
    }     // namespace op
} // namespace ngraph
