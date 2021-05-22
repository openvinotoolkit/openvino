// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/gather_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Gather slices from axis of params according to indices
            class NGRAPH_API Gather : public op::util::GatherBase
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                static const int64_t AXIS_NOT_SET_VALUE = std::numeric_limits<int64_t>::max();
                Gather() = default;
                /// \param params The tensor from which slices are gathered
                /// \param indices Tensor with indexes to gather
                /// \param axis The tensor is a dimension index to gather data from
                Gather(const Output<Node>& params,
                       const Output<Node>& indices,
                       const Output<Node>& axis);

                bool visit_attributes(AttributeVisitor& visitor) override;
                int64_t get_axis() const override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v1

        namespace v7
        {
            /// \brief Gather slices from axis of params according to indices
            class NGRAPH_API Gather : public op::util::GatherBase
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                Gather() = default;

                /// \param data The tensor from which slices are gathered
                /// \param indices Tensor with indexes to gather
                /// \param axis The tensor is a dimension index to gather data from
                /// \param batch_dims The number of batch dimension in data and indices tensors.
                /// If batch_dims = 0 Gather v7 is identical to Gather v1.
                Gather(const Output<Node>& data,
                       const Output<Node>& indices,
                       const Output<Node>& axis,
                       const int64_t batch_dims = 0);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;
                int64_t get_batch_dims() const;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v7
    }     // namespace op
} // namespace ngraph
