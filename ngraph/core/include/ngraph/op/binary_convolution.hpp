//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
        namespace v1
        {
            class NGRAPH_API BinaryConvolution : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"BinaryConvolution", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                enum class BinaryConvolutionMode
                {
                    // Interpret input data and kernel values: 0 as -1, 1 as 1
                    XNOR_POPCOUNT
                };

                /// \brief Constructs a binary convolution operation.
                BinaryConvolution() = default;
                /// \brief Constructs a binary convolution operation.
                /// \param data The node producing the input data batch tensor.
                /// \param kernel The node producing the filters tensor.
                /// \param strides The strides.
                /// \param pads_begin The beginning of padding shape.
                /// \param pads_end The end of padding shape.
                /// \param dilations The dilations.
                /// \param mode Defines how input tensor 0/1 values and weights 0/1 are interpreted.
                /// \param pad_value Floating-point value used to fill pad area.
                /// \param auto_pad The pad type for automatically computing padding sizes.
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                BinaryConvolution(const Output<Node>& data,
                                  const Output<Node>& kernel,
                                  const Strides& strides,
                                  const CoordinateDiff& pads_begin,
                                  const CoordinateDiff& pads_end,
                                  const Strides& dilations,
                                  BinaryConvolutionMode mode,
                                  float pad_value,
                                  const PadType& auto_pad = PadType::EXPLICIT);

                BinaryConvolution(const Output<Node>& data,
                                  const Output<Node>& kernel,
                                  const Strides& strides,
                                  const CoordinateDiff& pads_begin,
                                  const CoordinateDiff& pads_end,
                                  const Strides& dilations,
                                  const std::string& mode,
                                  float pad_value,
                                  const PadType& auto_pad = PadType::EXPLICIT);

                size_t get_version() const override { return 1; }
                void validate_and_infer_types() override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The strides.
                const Strides& get_strides() const { return m_strides; }
                void set_strides(const Strides& strides) { m_strides = strides; }
                /// \return The dilations.
                const Strides& get_dilations() const { return m_dilations; }
                void set_dilations(const Strides& dilations) { m_dilations = dilations; }
                /// \return The padding-below sizes (possibly negative).
                const CoordinateDiff& get_pads_begin() const { return m_pads_begin; }
                void set_pads_begin(const CoordinateDiff& pads_begin) { m_pads_begin = pads_begin; }
                /// \return The padding-above sizes (possibly negative).
                const CoordinateDiff& get_pads_end() const { return m_pads_end; }
                void set_adding_above(const CoordinateDiff& pads_end) { m_pads_end = pads_end; }
                /// \return The pad type for convolution.
                const PadType& get_auto_pad() const { return m_auto_pad; }
                void set_auto_pad(const PadType& auto_pad) { m_auto_pad = auto_pad; }
                /// \return The mode of convolution.
                const BinaryConvolutionMode& get_mode() const { return m_mode; }
                void set_mode(const BinaryConvolutionMode& mode) { m_mode = mode; }
                /// \return The pad value.
                float get_pad_value() const { return m_pad_value; }
                void set_pad_value(float pad_value) { m_pad_value = pad_value; }

            protected:
                BinaryConvolutionMode mode_from_string(const std::string& mode) const;
                Strides m_strides;
                Strides m_dilations;
                CoordinateDiff m_pads_begin;
                CoordinateDiff m_pads_end;
                BinaryConvolutionMode m_mode;
                float m_pad_value;
                PadType m_auto_pad;
            };
        }
    } // namespace op

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s,
                             const op::v1::BinaryConvolution::BinaryConvolutionMode& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v1::BinaryConvolution::BinaryConvolutionMode>
        : public EnumAttributeAdapterBase<op::v1::BinaryConvolution::BinaryConvolutionMode>
    {
    public:
        AttributeAdapter(op::v1::BinaryConvolution::BinaryConvolutionMode& value)
            : EnumAttributeAdapterBase<op::v1::BinaryConvolution::BinaryConvolutionMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v1::BinaryConvolution::BinaryConvolutionMode>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

} // namespace ngraph
