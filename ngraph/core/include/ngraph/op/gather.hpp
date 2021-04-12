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
            /// \brief Gather slices from axis of params according to indices
            class NGRAPH_API Gather : public Op
            {
            public:
                static const int64_t AXIS_NOT_SET_VALUE = std::numeric_limits<int64_t>::max();
                static constexpr NodeTypeInfo type_info{"Gather", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Gather() = default;
                /// \param params The tensor from which slices are gathered
                /// \param indices Tensor with indexes to gather
                /// \param axis The tensor is a dimension index to gather data from
                Gather(const Output<Node>& params,
                       const Output<Node>& indices,
                       const Output<Node>& axis);

                bool visit_attributes(AttributeVisitor& visitor) override;
                int64_t get_axis() const;

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool evaluate_lower(const HostTensorVector& outputs) const override;
                bool evaluate_upper(const HostTensorVector& outputs) const override;

                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& inputs_values) override;

            private:
                static const int PARAMS;
                static const int INDICES;
                static const int AXIS;

                bool evaluate_gather(const HostTensorVector& outputs,
                                     const HostTensorVector& inputs) const;
            };
        } // namespace v1

        namespace v7
        {
            /// \brief Gather slices from axis of params according to indices
            class NGRAPH_API Gather : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                Gather() = default;

                /// \param data The tensor from which slices are gathered
                /// \param indices Tensor with indexes to gather
                /// \param axis The tensor is a dimension index to gather data from
                /// \param batch_dims The number of batch dimension in data and indices tensors
                Gather(const Output<Node>& data,
                       const Output<Node>& indices,
                       const Output<Node>& axis,
                       const int64_t batch_dims = 0);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                int64_t get_batch_dims() const;
                int64_t get_axis() const;
                bool is_axis_set() const;

                bool evaluate_gather(const HostTensorVector& outputs,
                                     const HostTensorVector& inputs) const;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool evaluate_lower(const HostTensorVector& outputs) const override;
                bool evaluate_upper(const HostTensorVector& outputs) const override;

                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& inputs_values) override;

            private:
                int64_t m_batch_dims = 0;
            };
        } // namespace v7
    }     // namespace op
} // namespace ngraph
