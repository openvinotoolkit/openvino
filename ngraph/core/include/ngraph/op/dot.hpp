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

#include <utility>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Generalized dot product operation, including scalar-tensor product,
            /// matrix-vector
            ///        product, and matrix multiplication.
            class NGRAPH_API Dot : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Dot", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a dot product operation.
                Dot() = default;
                /// \brief Constructs a dot product operation.
                ///
                /// \param arg0 The node producing the first argument.
                /// \param arg1 The node producing the second argument.
                /// \param reduction_axes_count The number of axes to dot.
                Dot(const Output<Node>& arg0,
                    const Output<Node>& arg1,
                    size_t reduction_axes_count,
                    bool has_reduction_axes_count = true);

                /// \brief Constructs a dot product operation with default dot-axis selection
                /// depending
                ///        on the inputs.
                ///
                /// If `arg0` or `arg1` is a scalar, there are no dot-axes. Else, there is one
                /// dot-axis.
                ///
                /// (Note that in particular, this results in scalar-tensor products where one or
                /// the
                /// other argument is a scalar, a matrix-vector products where `arg0` is a matrix
                /// and
                /// `arg1` is a vector, and a matrix multiplication where `arg0` and `arg1` are both
                /// matrices.)
                ///
                /// \param arg0 The node producing the first argument.
                /// \param arg1 The node producing the second argument.
                Dot(const Output<Node>& arg0, const Output<Node>& arg1);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node> get_default_value() const override;

                size_t get_reduction_axes_count() const { return m_reduction_axes_count; }
                void set_reduction_axes_count(size_t reduction_axes_count)
                {
                    m_reduction_axes_count = reduction_axes_count;
                }
                bool get_has_reduction_axes_count() const { return m_has_reduction_axes_count; }
                void set_has_reduction_axes_count(bool has_reduction_axes_count)
                {
                    m_has_reduction_axes_count = has_reduction_axes_count;
                }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override
                {
                    check_new_args_count(this, new_args);
                    return std::make_shared<Dot>(
                        new_args.at(0), new_args.at(1), m_reduction_axes_count);
                }

            protected:
                size_t m_reduction_axes_count;
                bool m_has_reduction_axes_count;
            };
        }
        using v0::Dot;
    }
}
