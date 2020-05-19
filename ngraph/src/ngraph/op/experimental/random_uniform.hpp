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
            /// \brief Generates a tensor populated with random values of a uniform distribution.
            class NGRAPH_API RandomUniform : public op::Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"RandomUniform", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs an uninitialized RandomUniform node.
                RandomUniform() = default;

                /// \brief Constructs a RandomUniform node.
                /// \param min_value Output producing the minimum value (inclusive) for the random
                ///                  uniform distribution. Must return a scalar of floating point
                ///                  type, and the type must match that of `max_value`.
                /// \param max_value Output producing the maximum value (inclusive) for the random
                ///                  uniform distribution. Must return a scalar of floating point
                ///                  type, and the type must match that of `min_value`.
                /// \param result_shape Output producing the shape of the output tensor. Must return
                ///                     a vector of type `element::i64`.
                /// \param use_fixed_seed Output producing a boolean scalar Flag indicating whether
                ///                       to use the value supplied in `fixed_seed` to re-seed the
                ///                       random number generator at this iteration. Note that
                ///                       whenever `use_fixed_seed` is `true`, the same values will
                ///                       be generated in the output tensor. This flag is primarily
                ///                       used for debugging. If `use_fixed_seed` is `false`, the
                ///                       value in `fixed_seed` is ignored.
                /// \param fixed_seed Fixed seed value to be supplied to the random number generator
                ///                   if `use_fixed_seed` is `true`. If `use_fixed_seed` is `false`,
                ///                   this value is ignored.
                RandomUniform(const Output<Node>& min_value,
                              const Output<Node>& max_value,
                              const Output<Node>& result_shape,
                              const Output<Node>& use_fixed_seed,
                              uint64_t fixed_seed);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \brief Returns the fixed seed value to be supplied to the random number
                ///        generator if `use_fixed_seed` is `true`. If `use_fixed_seed` is `false`,
                ///        this value is ignored.
                uint64_t get_fixed_seed() const { return m_fixed_seed; }
                /// \brief Sets the fixed seed value to be supplied to the random number generator
                ///        if `use_fixed_seed` is `true`. If `use_fixed_seed` is `false`, this value
                ///        is ignored.
                void set_fixed_seed(uint64_t fixed_seed) { m_fixed_seed = fixed_seed; }
                // Internally, any implementation of RandomUniform will have state, since it is
                // backed by a random number generator.
                bool has_state() const override { return true; }
                void validate_and_infer_types() override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                               const OutputVector& /* deltas */) override
                {
                }
                uint64_t m_fixed_seed;
            };
        }
        using v0::RandomUniform;
    }
}
