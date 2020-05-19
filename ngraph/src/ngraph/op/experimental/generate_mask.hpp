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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief GenerateMask
            ///
            class NGRAPH_API GenerateMask : public op::Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"GenerateMask", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a GenerateMask node with a given shape, seed,
                /// probability and training/inference mode
                GenerateMask() = default;

                /// \brief Constructs a GenerateMask node with a given shape, seed,
                /// probability and training/inference mode
                GenerateMask(const Output<Node>& training,
                             const Shape& shape,
                             const element::Type& element_type,
                             uint64_t seed,
                             double prob,
                             bool use_seed = false);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const element::Type& get_element_type() const { return m_element_type; }
                void set_element_type(const element::Type& element_type)
                {
                    m_element_type = element_type;
                }
                /// Deprecated accessor for transitional attributes
                const Shape& get_mask_shape() const { return m_shape; }
                /// \brief Returns the probability of a trial generating 1 (i.e. an element being
                /// kept)
                double get_probability() const { return m_probability; }
                /// \brief Returns the seed value supplied to a random generator
                uint64_t get_seed() const { return m_seed; }
                bool get_use_seed() const { return m_use_seed; }
                /// GenerateMask has state.
                bool has_state() const override { return true; }
                void validate_and_infer_types() override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                               const OutputVector& /* deltas */) override
                {
                }
                element::Type m_element_type;
                // These will be deprecated
                Shape m_shape;
                bool m_use_seed{false};
                uint64_t m_seed{0};
                double m_probability{0.0};
            };
        } // namespace v0

        namespace v1
        {
            /// \brief GenerateMask
            ///
            class NGRAPH_API GenerateMask : public op::Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"GenerateMask", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a GenerateMask node with a given shape, seed,
                /// probability and training/inference mode
                GenerateMask() = default;

                /// \brief Constructs a GenerateMask node with a given shape, seed,
                /// probability and training/inference mode
                GenerateMask(const Output<Node>& training,
                             const Output<Node>& shape,
                             const element::Type& element_type,
                             uint64_t seed,
                             double prob,
                             bool use_seed = false);

                size_t get_version() const override { return 1; }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const element::Type& get_element_type() const { return m_element_type; }
                void set_element_type(const element::Type& element_type)
                {
                    m_element_type = element_type;
                }
                /// Deprecated accessor for transitional attributes
                const Shape get_mask_shape() const;
                /// \brief Returns the probability of a trial generating 1 (i.e. an element being
                /// kept)
                double get_probability() const { return m_probability; }
                /// \brief Returns the seed value supplied to a random generator
                uint64_t get_seed() const { return m_seed; }
                bool get_use_seed() const { return m_use_seed; }
                /// GenerateMask has state.
                bool has_state() const override { return true; }
                void validate_and_infer_types() override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                               const OutputVector& /* deltas */) override
                {
                }
                element::Type m_element_type;
                // These will be deprecated
                bool m_use_seed{false};
                uint64_t m_seed{0};
                double m_probability{0.0};
            };
        } // namespace v1

        using v0::GenerateMask;
    } // op
} // ngraph
