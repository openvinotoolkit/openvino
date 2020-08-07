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

#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief SpaceToDepth permutes input tensor blocks of spatial data into depth
            /// dimension.
            ///
            /// \note  Values from the height and width dimensions are moved to the depth dimension.
            ///
            ///        Output node produces a tensor with shape:
            ///        [N, C * blocksize * blocksize, H / blocksize, W / blocksize]
            class NGRAPH_API SpaceToDepth : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"SpaceToDepth", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                enum class SpaceToDepthMode
                {
                    // The output depth is gathered from [block_size, ..., block_size, C]
                    BLOCKS_FIRST,
                    // The output depth is gathered from [C, block_size, ..., block_size]
                    DEPTH_FIRST
                };

                SpaceToDepth() = default;
                /// \brief Constructs a SpaceToDepth operation.
                ///
                /// \param data - Node producing the input tensor
                /// \param mode Specifies how the output depth dimension is gathered
                /// from block coordinates and the old depth dimension.
                /// \param block_size - the size of the block of values to be moved
                SpaceToDepth(const Output<Node>& data,
                             const SpaceToDepthMode& mode,
                             std::size_t block_size = 1);

                SpaceToDepth(const Output<Node>& data,
                             const std::string& mode,
                             std::size_t block_size = 1);

                bool visit_attributes(AttributeVisitor& visitor) override;
                std::size_t get_block_size() const { return m_blocksize; }
                SpaceToDepthMode get_mode() const { return m_mode; }
                virtual OutputVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                std::size_t m_blocksize;
                SpaceToDepthMode m_mode;
            };
        }
        using v0::SpaceToDepth;
    } // namespace op

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const op::v0::SpaceToDepth::SpaceToDepthMode& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v0::SpaceToDepth::SpaceToDepthMode>
        : public EnumAttributeAdapterBase<op::v0::SpaceToDepth::SpaceToDepthMode>
    {
    public:
        AttributeAdapter(op::v0::SpaceToDepth::SpaceToDepthMode& value)
            : EnumAttributeAdapterBase<op::v0::SpaceToDepth::SpaceToDepthMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v0::SpaceToDepth::SpaceToDepthMode>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
