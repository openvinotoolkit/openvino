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
        namespace v0
        {
            /// \brief Takes a slice of an input tensor, i.e., the sub-tensor that resides within a
            ///        bounding box, optionally with stride.
            class NGRAPH_DEPRECATED(
                "This operation is deprecated and will be removed soon. Please do not use it.")
                NGRAPH_API Slice : public Op
            {
                NGRAPH_SUPPRESS_DEPRECATED_START
            public:
                static constexpr NodeTypeInfo type_info{"Slice", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a tensor slice operation
                Slice() = default;
                /// \brief Constructs a tensor slice operation.
                ///
                /// \param arg The tensor to be sliced.
                /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
                /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
                /// \param strides The slicing strides; for example, strides of `{n,m}` means to
                /// take
                ///                every nth row and every mth column of the input matrix.
                Slice(const Output<Node>& arg,
                      const Coordinate& lower_bounds,
                      const Coordinate& upper_bounds,
                      const Strides& strides);
                /// \brief Constructs a tensor slice operation with unit strides; i.e., every
                /// element
                ///        inside the bounding box will be copied to the output slice.
                ///
                /// \param arg The tensor to be sliced.
                /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
                /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
                Slice(const Output<Node>& arg,
                      const Coordinate& lower_bounds,
                      const Coordinate& upper_bounds);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                void validate_and_infer_types() override;

                /// \return The inclusive lower-bound coordinates.
                const Coordinate& get_lower_bounds() const { return m_lower_bounds; }
                /// \return The exclusive upper-bound coordinates.
                const Coordinate& get_upper_bounds() const { return m_upper_bounds; }
                /// \return The slicing strides.
                const Strides& get_strides() const { return m_strides; }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

            protected:
                Coordinate m_lower_bounds;
                Coordinate m_upper_bounds;
                Strides m_strides;
                NGRAPH_SUPPRESS_DEPRECATED_END
            };
        }
        // default opset version
        NGRAPH_SUPPRESS_DEPRECATED_START
        using v0::Slice;
        NGRAPH_SUPPRESS_DEPRECATED_END
    }
}
