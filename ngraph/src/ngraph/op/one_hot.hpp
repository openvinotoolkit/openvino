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
        namespace v0
        {
            // clang-format off
            /// \brief One-hot operator.
            ///
            /// ## Parameters
            ///
            /// |                | Description                                                |
            /// | -------------- | ---------------------------------------------------------- |
            /// | `shape`        | The desired output shape, including the new one-hot axis.  |
            /// | `one_hot_axis` | The index within the output shape of the new one-hot axis. |
            ///
            /// ## Inputs
            ///
            /// |       | Type                                                    | Description                                                    |
            /// | ----- | ------------------------------------------------------- | -------------------------------------------------------------- |
            /// | `arg` | \f$E[d_1,\dots,d_{m-1},d_{m+1},\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and any non-floating point element type. |
            ///
            /// ## Output
            ///
            /// | Type                   | Description                                                                                                                                                                                                                                                                |
            /// | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
            /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T'\f$, where \f$T'[i_1,\dots,i_{m-1},i_m,i_{m+1},\dots,i_n] = 1\f$ if \f$T[i_1,\dots,i_{m-1},i_{m+1},\dots,i_n] = i_m\f$, else \f$0\f$. However, \f$T'\f$ is undefined if any non-integral value or any out-of-bounds value is detected in the input tensor. |
            // clang-format on
            class NGRAPH_API OneHot : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"OneHot", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a one-hot operation.
                OneHot() = default;
                /// \brief Constructs a one-hot operation.
                ///
                /// \param arg          Node that produces the input tensor to be one-hot encoded.
                /// \param shape        The shape of the output tensor, including the new one-hot
                /// axis.
                /// \param one_hot_axis The index within the output shape of the new one-hot axis.
                OneHot(const Output<Node>& arg, const PartialShape& shape, size_t one_hot_axis);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                void validate_and_infer_types() override;

                /// \return The index of the one-hot axis.
                size_t get_one_hot_axis() const { return m_one_hot_axis; }
                void set_one_hot_axis(size_t one_hot_axis) { m_one_hot_axis = one_hot_axis; }
            protected:
                PartialShape m_shape;
                size_t m_one_hot_axis;
            };
        }
        namespace v1
        {
            class NGRAPH_API OneHot : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"OneHot", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a one-hot operation.
                OneHot() = default;
                /// \brief Constructs a one-hot operation.
                ///
                /// \param indices   Input tensor containing indices.
                /// \param depth     Specifies number of classes and the size of one-hot dimension.
                /// \param on_value  Specifies value that the locations in output tensor represented
                ///                  by indices in input take.
                /// \param off_value Specifies value that the locations in output tensor not
                /// represented
                ///                  by indices in input take.
                /// \param axis      Axis along which one-hot representation in added.
                OneHot(const Output<Node>& indices,
                       const Output<Node>& depth,
                       const Output<Node>& on_value,
                       const Output<Node>& off_value,
                       int64_t axis);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                void validate_and_infer_types() override;

                /// \return The index of the one-hot axis.
                int64_t get_axis() const { return m_axis; }
                void set_axis(int64_t axis) { m_axis = axis; }
            protected:
                int64_t m_axis;
            };
        }
        // default opset version
        using v0::OneHot;
    }
}
