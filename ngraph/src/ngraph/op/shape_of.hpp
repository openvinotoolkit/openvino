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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v3
        {
            /// \brief Operation that returns the shape of its input argument as a tensor.
            class NGRAPH_API ShapeOf : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ShapeOf", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ShapeOf() = default;
                /// \brief Constructs a shape-of operation.
                ShapeOf(const Output<Node>& arg, const element::Type output_type = element::i64);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                void validate_and_infer_types() override;

                element::Type get_output_type() const { return m_output_type; }
                void set_output_type(element::Type output_type) { m_output_type = output_type; }
                // Overload collision with method on Node
                using Node::set_output_type;

                // FOR CONSTANT FOLDING INTERNAL USAGE ONLY
                // Constant folding for cases with static rank but dynamic shape create a subgraph
                // which contains a Shape of.
                // In this case we need to prevent constant folding from endless creation of these
                // subgraphs.
                // These metods should be removed if better solution will be designed.
                void set_is_foldable(bool is_foldable) { m_is_foldable = is_foldable; }
                bool get_is_foldable() const { return m_is_foldable; }
                bool evaluate(const HostTensorVector& output_values,
                              const HostTensorVector& input_values) const override;
                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& input_values) override;

            private:
                bool m_is_foldable = true;
                element::Type m_output_type;
            };
        }

        namespace v0
        {
            /// \brief Operation that returns the shape of its input argument as a tensor.
            class NGRAPH_API ShapeOf : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ShapeOf", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ShapeOf() = default;
                /// \brief Constructs a shape-of operation.
                ShapeOf(const Output<Node>& arg);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                void validate_and_infer_types() override;

                // FOR CONSTANT FOLDING INTERNAL USAGE ONLY
                // Constant folding for cases with static rank but dynamic shape create a subgraph
                // which contains a Shape of.
                // In this case we need to prevent constant folding from endless creation of these
                // subgraphs.
                // These metods should be removed if better solution will be designed.
                void set_is_foldable(bool is_foldable) { m_is_foldable = is_foldable; }
                bool get_is_foldable() const { return m_is_foldable; }
                bool evaluate(const HostTensorVector& output_values,
                              const HostTensorVector& input_values) const override;
                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& input_values) override;

            private:
                bool m_is_foldable = true;
            };
        }
        using v0::ShapeOf;
    }
}
