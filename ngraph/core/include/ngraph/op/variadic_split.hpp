// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief VariadicSplit operation splits an input tensor into pieces along some axis.
            /// The pieces may have variadic lengths depending on "split_lengths" attribute.
            class NGRAPH_API VariadicSplit : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs a variadic split operation.
                VariadicSplit() = default;
                /// \brief Constructs a variadic split operation.
                ///
                /// \param data           The tensor to be split.
                /// \param axis           The index of an axis in "data" along which to perform the
                /// split.
                /// \param split_lengths  A list containing the sizes of each output tensor
                /// along the split "axis". Size of "split_lengths" should be equal to the number of
                ///
                /// outputs. The sum of split_lengths must match data.shape[axis]
                VariadicSplit(const Output<Node>& data,
                              const Output<Node>& axis,
                              const Output<Node>& split_lengths);

                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                size_t get_default_output_index() const override { return no_default_index(); }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            private:
                bool evaluate_variadic_split(const HostTensorVector& outputs,
                                             const HostTensorVector& inputs) const;
            };
        } // namespace v1

        using v1::VariadicSplit;
    } // namespace op
} // namespace ngraph
