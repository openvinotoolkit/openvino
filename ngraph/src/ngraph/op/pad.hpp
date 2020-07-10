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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Generic padding operation.
            class NGRAPH_API Pad : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Pad", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a generic padding operation.
                Pad() = default;
                /// \brief Constructs a padding operation. Padding embeds the values of the input
                /// tensor into a larger tensor initialized to arg_pad_value.
                ///
                /// \param arg The node producing the input tensor to be padded.
                /// \param arg_pad_value The node producing the scalar value
                /// to be used outside the are initialized by arg when pad_mode is CONSTANT.
                /// \param padding_below How many elements to add on
                /// each axis before index 0 of arg. Rank must match arg.
                /// \param padding_above How many elements to add on
                /// each axis after the last element of arg. Rank must match arg.
                /// \param pad_mode The padding mode: CONSTANT(default), EDGE, REFLECT or SYMMETRIC.
                /// CONSTANT initializes new elements with arg_pad_value, EDGE uses the nearest
                /// value from arg. REFLECT and SYMMETRIC tile the background by flipping arg
                /// at the edge (SYMMETRIC) or on the last row/column/etc. (REFLECT).
                Pad(const Output<Node>& arg,
                    const Output<Node>& arg_pad_value,
                    const CoordinateDiff& padding_below,
                    const CoordinateDiff& padding_above,
                    PadMode pad_mode = PadMode::CONSTANT);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                void validate_and_infer_types() override;
                /// \return The padding-below sizes.
                const CoordinateDiff& get_padding_below() const { return m_padding_below; }
                void set_padding_below(const CoordinateDiff& padding_below)
                {
                    m_padding_below = padding_below;
                }
                /// \return The padding-above sizes.
                const CoordinateDiff& get_padding_above() const { return m_padding_above; }
                void set_padding_above(const CoordinateDiff& padding_above)
                {
                    m_padding_above = padding_above;
                }
                /// \brief DEPRECATED. This is just a stub for backends that used to implement the
                ///        interior padding feature, which is no longer supported.
                /// \return Returns a shape full of zeros,
                /// with the same rank as get_padding_below().
                const Shape& get_padding_interior() const { return m_padding_interior_fake; }
                /// \return The padding mode.
                PadMode get_pad_mode() const { return m_pad_mode; }
                void set_pad_mode(PadMode pad_mode) { m_pad_mode = pad_mode; }
                /// \return The default value for Pad.
                virtual std::shared_ptr<Node> get_default_value() const override;

            protected:
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                Shape m_padding_interior_fake; // LEGACY: This is all zeros.
                PadMode m_pad_mode;
            };
        }

        namespace v1
        {
            /// \brief Generic padding operation.
            class NGRAPH_API Pad : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Pad", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a generic padding operation.
                ///
                /// \param arg The output producing input tensor to be padded.
                /// \param pads_begin The output which specifies the number of padding elements
                /// added
                /// before position 0 on each axis of arg.
                /// \param pads_end The output which specifies the number of padding elements
                /// after the last element on each axis.
                /// \param arg_pad_value The scalar output with the value used for padding
                /// if pad_mode is CONSTANT
                /// \param pad_mode The padding mode: CONSTANT, EDGE, REFLECT or SYMMETRIC.
                /// CONSTANT initializes new elements with arg_pad_value, EDGE uses the nearest
                /// value from arg. REFLECT and SYMMETRIC tile the background by flipping arg
                /// at the edge (SYMMETRIC) or on the last row/column/etc. (REFLECT).
                Pad(const Output<Node>& arg,
                    const Output<Node>& pads_begin,
                    const Output<Node>& pads_end,
                    const Output<Node>& arg_pad_value,
                    PadMode pad_mode);

                /// \brief Constructs a generic padding operation.
                ///
                /// \param arg The output producing input tensor to be padded.
                /// \param pads_begin The output which specifies the number of padding elements
                /// added
                /// \param pads_end The output which specifies the number of padding elements
                /// after the last element on each axis.
                /// \param pad_mode The padding mode: CONSTANT, EDGE, REFLECT or SYMMETRIC.
                Pad(const Output<Node>& arg,
                    const Output<Node>& pads_begin,
                    const Output<Node>& pads_end,
                    PadMode pad_mode);

                /// \brief Constructs a generic padding operation.
                Pad() = default;

                bool visit_attributes(AttributeVisitor& visitor) override;
                size_t get_version() const override { return 1; }
                void validate_and_infer_types() override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// return The node which specifies the number of padding elements
                /// added at the beginning of each axis
                CoordinateDiff get_pads_begin() const;
                /// return The node which specifies the number of padding elements
                /// added at the end of each axis
                CoordinateDiff get_pads_end() const;

                /// \return The padding mode.
                PadMode get_pad_mode() const { return m_pad_mode; }
                void set_pad_mode(PadMode pad_mode) { m_pad_mode = pad_mode; }
            private:
                PadMode m_pad_mode;
            };
        }

        // latest stable opset version
        using v0::Pad;
    }
}
