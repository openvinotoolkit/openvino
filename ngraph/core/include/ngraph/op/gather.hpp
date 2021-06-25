// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v8
        {
            /// \brief Gather slices from axis of params according to indices. This is general
            /// solution which supports negative indices
            class NGRAPH_API Gather : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                Gather() = default;

                /// \param data The tensor from which slices are gathered
                /// \param indices Tensor with indexes to gather (negative values are allowed)
                /// \param axis The tensor is a dimension index to gather data from
                /// \param batch_dims The number of batch dimension in data and indices tensors
                /// If batch_dims = 0 Gather v7 is identical to Gather v1.
                Gather(const Output<Node>& data,
                       const Output<Node>& indices,
                       const Output<Node>& axis,
                       const int64_t batch_dims = 0);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& inputs_values) override;

                bool evaluate_lower(const HostTensorVector& outputs) const override;

                bool evaluate_upper(const HostTensorVector& outputs) const override;
                int64_t get_batch_dims() const;
                virtual int64_t get_axis() const;
            protected:
                int64_t m_batch_dims = 0;
                void validate();
                void infer_partial_shape_and_types();
            };
        } // namespace v8

        namespace v7
        {
            /// \brief Gather slices from axis of params according to indices
            class NGRAPH_API Gather : public op::v8::Gather
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                Gather() = default;

                /// \param data The tensor from which slices are gathered
                /// \param indices Tensor with indexes to gather
                /// \param axis The tensor is a dimension index to gather data from
                /// \param batch_dims The number of batch dimension in data and indices tensors.
                Gather(const Output<Node>& data,
                       const Output<Node>& indices,
                       const Output<Node>& axis,
                       const int64_t batch_dims = 0);

                std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v7

        namespace v1
        {
            /// \brief Gather slices from axis of params according to indices
            class NGRAPH_API Gather : public op::v7::Gather
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
                void validate_and_infer_types() override;
                int64_t get_axis() const override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v1

    } // namespace op
} // namespace ngraph
