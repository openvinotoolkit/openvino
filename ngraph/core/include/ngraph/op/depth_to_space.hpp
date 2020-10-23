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
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fused_op.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief DepthToSpace permutes data from the depth dimension of the input blob into
            ///        spatial dimensions.
            ///
            /// \note  Values from the depth dimension (assuming NCHW layout) are moved in
            ///        spatial blocks to the height and width dimensions.
            ///
            ///        Output node produces a tensor with shape:
            ///        [N, C/(blocksize * blocksize), H * blocksize, W * blocksize]
            class NGRAPH_API DepthToSpace : public ngraph::op::util::FusedOp
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                enum class DepthToSpaceMode
                {
                    // The input depth is divided to [block_size, ..., block_size, new_depth]
                    BLOCKS_FIRST,
                    // The input depth is divided to [new_depth, block_size, ..., block_size]
                    DEPTH_FIRST
                };

                DepthToSpace() = default;
                /// \brief Constructs a DepthToSpace operation.
                ///
                /// \param data Node producing the input tensor
                /// \param mode Specifies how the input depth dimension is split to block
                /// coordinates
                /// \param block_size The size of the block of values to be moved
                DepthToSpace(const Output<Node>& data,
                             const DepthToSpaceMode& mode,
                             std::size_t block_size = 1);

                DepthToSpace(const Output<Node>& data,
                             const std::string& mode,
                             std::size_t block_size = 1);
                bool visit_attributes(AttributeVisitor& visitor) override;

                std::size_t get_block_size() const { return m_blocksize; }
                DepthToSpaceMode get_mode() const { return m_mode; }
                virtual OutputVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                std::size_t m_blocksize;
                DepthToSpaceMode m_mode;
                DepthToSpaceMode mode_from_string(const std::string& mode) const;
            };
        }
        using v0::DepthToSpace;
    }

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const op::v0::DepthToSpace::DepthToSpaceMode& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v0::DepthToSpace::DepthToSpaceMode>
        : public EnumAttributeAdapterBase<op::v0::DepthToSpace::DepthToSpaceMode>
    {
    public:
        AttributeAdapter(op::v0::DepthToSpace::DepthToSpaceMode& value)
            : EnumAttributeAdapterBase<op::v0::DepthToSpace::DepthToSpaceMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v0::DepthToSpace::DepthToSpaceMode>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
}

NGRAPH_SUPPRESS_DEPRECATED_END
