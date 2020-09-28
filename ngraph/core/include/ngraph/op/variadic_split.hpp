//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
                static constexpr NodeTypeInfo type_info{"VariadicSplit", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
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
            };
        } // namespace v1

        using v1::VariadicSplit;
    } // namespace op
} // namespace ngraph
