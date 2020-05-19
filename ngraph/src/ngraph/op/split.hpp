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
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Splits the input tensor into a list of smaller tensors ("pieces")
            class NGRAPH_API Split : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"Split", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Split() = default;
                /// \brief Constructs a Split op that evenly divides the input tensor.
                ///
                /// \param data       Node producing the input tensor
                /// \param axis       Node producing an axis along which the input tensor
                ///                   should be split. Negative values mean counting from
                ///                   the back of the input tensor's shape.
                /// \param num_split  a number of "pieces" the input tensor will be split to
                Split(const Output<Node>& data, const Output<Node>& axis, const size_t num_split);

                /// \brief Constructs a Split op that splits the input tensor into variable length
                ///        "pieces"
                ///
                /// \param data    Node producing the input tensor
                /// \param axis    Node producing an axis along which the input tensor
                ///                should be split. Negative values mean counting from
                ///                the back of the input tensor's shape.
                /// \param splits  a list of lengths that the input tensor should be
                ///                split to. Use this constructor to split the input
                ///                tensor to variable length chunks.
                Split(const Output<Node>& data,
                      const Output<Node>& axis,
                      const std::vector<size_t>& splits);

                void pre_validate_and_infer_types() override;

                virtual NodeVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_axis() const { return m_axis; }
                const std::vector<size_t>& get_splits() const { return m_splits; }
            private:
                /// used internally for validation purposes, indicates which constructor was used
                bool m_split_evenly;
                int64_t m_axis;
                size_t m_num_split;
                /// contains lengths of chunks that the input tensor will be split into
                std::vector<size_t> m_splits;
            };
        }

        namespace v1
        {
            /// \brief Splits the input tensor into a list of equal sized tensors
            class NGRAPH_API Split : public ngraph::op::Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Split", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
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
            protected:
                size_t m_num_splits;
            };
        }

        using v0::Split;
    }
}
