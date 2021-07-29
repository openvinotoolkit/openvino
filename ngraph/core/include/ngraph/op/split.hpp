// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Splits the input tensor into a list of equal sized tensors
            class NGRAPH_API Split : public ngraph::op::Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs a split operation.
                Split() = default;
                /// \brief Constructs a split operation.
                /// \param data        The tensor to be split.
                /// \param axis        The index of an axis in "data" along which to perform
                ///                    the split.
                /// \param num_splits  The number of pieces that the data tensor should be
                ///                    split into.
                Split(const Output<Node>& data, const Output<Node>& axis, const size_t num_splits);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_num_splits() const { return m_num_splits; }
                void set_num_splits(const size_t num_splits) { m_num_splits = num_splits; }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            protected:
                size_t m_num_splits;
            };
        } // namespace v1
    }     // namespace op
} // namespace ngraph
