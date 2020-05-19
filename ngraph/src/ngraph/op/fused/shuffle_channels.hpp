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

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Permutes data in the channel dimension of the input
            class NGRAPH_API ShuffleChannels : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"ShuffleChannels", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ShuffleChannels() = default;
                /// \brief Constructs a ShuffleChannels node.
                ///
                /// \param data - Node producing the input tensor
                /// \param axis - channel dimension index in the data tensor. A negative value means
                ///               that the index should be calculated from the back of the input
                ///               data
                ///               shape.
                /// \param groups - number of groups the channel dimension specified by axis should
                /// be
                ///                 split into
                ShuffleChannels(const Output<Node>& data,
                                const int axis = 1,
                                const size_t groups = 1UL);

                bool visit_attributes(AttributeVisitor& visitor) override;
                size_t get_zero_based_axis() const;

                virtual void pre_validate_and_infer_types() override;

                virtual NodeVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                int get_axis() const { return m_axis; }
                size_t get_groups() const { return m_groups; }
            private:
                /// \brief Generates a shape required to permute the data
                ///
                /// \param data_shape - Shape of the original input data tensor
                /// \return A 4D tensor to be used to reshape the input data before shuffling it
                Shape get_pre_shuffle_shape(const Shape& data_shape) const;

                int m_axis;
                size_t m_groups;
            };
        }
        using v0::ShuffleChannels;
    }
}
