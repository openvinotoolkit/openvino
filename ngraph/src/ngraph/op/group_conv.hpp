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

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Batched convolution operation, with optional window dilation and stride.
            class NGRAPH_API GroupConvolution : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"GroupConvolution", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched convolution operation.
                GroupConvolution() = default;
                /// \brief Constructs a batched convolution operation.
                ///
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                /// \param strides The strides.<br>
                /// `[f]`
                /// \param dilations The dilations.<br>
                /// `[f]`
                /// \param pads_begin The beginning of padding shape.<br>
                /// `[f]`
                /// \param pads_end The end of padding shape.<br>
                /// `[f]`
                /// \param auto_pad The pad type for automatically computing padding sizes.<br>
                /// `[f]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                GroupConvolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& strides,
                                 const CoordinateDiff& pads_begin,
                                 const CoordinateDiff& pads_end,
                                 const Strides& dilations,
                                 const PadType& auto_pad = PadType::EXPLICIT);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const OutputVector& deltas) override;

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
                /// \return The default value for Convolution.
                virtual std::shared_ptr<Node> get_default_value() const override;

                bool evaluate(const HostTensorVector &output_values, const HostTensorVector &input_values) override;

            protected:
                Strides m_strides;
                Strides m_dilations;
                CoordinateDiff m_pads_begin;
                CoordinateDiff m_pads_end;
                PadType m_auto_pad;
            };

            /// \brief Data batch backprop for batched convolution operation.
            class NGRAPH_API GroupConvolutionBackpropData : public op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"GroupConvolutionBackpropData", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched-convolution data batch-backprop operation.
                GroupConvolutionBackpropData() = default;
                // clang-format off
                //
                // \brief      Constructs a batched-convolution data batch-backprop operation.
                //
                // \param      data            The node producing data from forward-prop. Shape: [N,
                //                             C_INPUT * GROUPS, X1, ..., XD].
                // \param      filter          The node producing the filter from forward-prop. Shape:
                //                             [GROUPS, C_INPUT, C_OUTPUT, K_D, ..., K_1]
                // \param      output_shape    The shape of the data batch from forward-prop. It's size
                //                             should be equal to number of data spatial dimensions.
                // \param      strides         The strides from forward-prop.
                // \param      pads_begin      The padding-below sizes from forward-prop.
                // \param      pads_end        The padding-above sizes from forward-prop.
                // \param      dilations       The dilations from forward-prop.
                // \param      auto_pad        The pad type for automatically computing padding sizes.
                // \param      output_padding  The output padding adds additional amount of paddings per
                //                             each spatial axis in the output tensor.
                //
                // clang-format on
                //
                GroupConvolutionBackpropData(const Output<Node>& data,
                                             const Output<Node>& filter,
                                             const Output<Node>& output_shape,
                                             const Strides& strides,
                                             const CoordinateDiff& pads_begin,
                                             const CoordinateDiff& pads_end,
                                             const Strides& dilations,
                                             const PadType& auto_pad = PadType::EXPLICIT,
                                             const CoordinateDiff& output_padding = {});

                // clang-format off
                //
                // \brief      Constructs a batched-convolution data batch-backprop operation.
                //
                // \param      data            The node producing data from forward-prop. Shape: [N,
                //                             C_INPUT * GROUPS, X1, ..., XD].
                // \param      filter          The node producing the filter from forward-prop. Shape:
                //                             [GROUPS, C_INPUT, C_OUTPUT, K_D, ..., K_1]
                // \param      output_shape    The shape of the data batch from forward-prop. It's size
                //                             should be equal to number of data spatial dimensions.
                // \param      strides         The strides from forward-prop.
                // \param      dilations       The dilations from forward-prop.
                // \param      auto_pad        The pad type for automatically computing padding sizes.
                // \param      output_padding  The output padding adds additional amount of paddings per
                //                             each spatial axis in the output tensor.
                //
                // clang-format on
                //
                GroupConvolutionBackpropData(const Output<Node>& data,
                                             const Output<Node>& filter,
                                             const Output<Node>& output_shape,
                                             const Strides& strides,
                                             const Strides& dilations,
                                             const PadType& auto_pad,
                                             const CoordinateDiff& output_padding = {});

                // clang-format off
                //
                // \brief      Constructs a batched-convolution data batch-backprop operation.
                //
                // \param      data            The node producing data from forward-prop. Shape:
                //                             [N, C_INPUT * GROUPS, X1, ..., XD].
                // \param      filter          The node producing the filter from forward-prop. Shape:
                //                             [GROUPS, C_INPUT, C_OUTPUT, K_D, ..., K_1]
                // \param      strides         The strides from forward-prop.
                // \param      pads_begin      The padding-below sizes from forward-prop.
                // \param      pads_end        The padding-above sizes from forward-prop.
                // \param      dilations       The dilations from forward-prop.
                // \param      auto_pad        The pad type for automatically computing padding sizes.
                // \param      output_padding  The output padding adds additional amount of paddings per
                //                             each spatial axis in the output tensor.
                //
                // clang-format on
                GroupConvolutionBackpropData(const Output<Node>& data,
                                             const Output<Node>& filter,
                                             const Strides& strides,
                                             const CoordinateDiff& pads_begin,
                                             const CoordinateDiff& pads_end,
                                             const Strides& dilations,
                                             const PadType& auto_pad = PadType::EXPLICIT,
                                             const CoordinateDiff& output_padding = {});
                ///
                /// \brief      Calculates output spatial features size.
                ///
                /// \param[in]  input_data_shape      The input data partial shape
                /// \param[in]  filters_shape         The filters partial shape
                /// \param[in]  strides               The strides values.
                /// \param[in]  dilations             The dilations values.
                /// \param[in]  pads_begin            The paddings at the beginning of axis.
                /// \param[in]  pads_end              The paddings at the end of axis.
                /// \param[in]  output_padding    The output padding values.
                /// \param      output_spatial_shape  The placeholder for computed output spatial
                /// partial
                /// shape.
                ///
                void infer_conv_backprop_output_spatial_shape(
                    const std::vector<Dimension>& input_data_shape,
                    const std::vector<Dimension>& filters_shape,
                    const Strides& strides,
                    const Strides& dilations,
                    const CoordinateDiff& pads_begin,
                    const CoordinateDiff& pads_end,
                    const CoordinateDiff& output_padding,
                    std::vector<Dimension>& output_spatial_shape);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual bool is_dynamic() const override;
                virtual NodeVector decompose_op() const override;
                virtual void pre_validate_and_infer_types() override;

                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const OutputVector& deltas) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The spatial shape of the output.
                const PartialShape get_convolution_output_shape() const;
                void set_output_shape(const Shape& output_shape);
                /// \return The strides from the forward prop.
                const Strides& get_strides() const { return m_strides; }
                void set_strides(const Strides& strides) { m_strides = strides; }
                /// \return The dilations from the forward prop.
                const Strides& get_dilations() const { return m_dilations; }
                void set_dilations(const Strides& dilations) { m_dilations = dilations; }
                /// \return The number of pixels to add to the beginning along each axis.
                const CoordinateDiff& get_pads_begin() const { return m_pads_begin; }
                void set_pads_begin(const CoordinateDiff& pads_begin) { m_pads_begin = pads_begin; }
                /// \return The number of pixels to add to the ending along each axis.
                const CoordinateDiff& get_pads_end() const { return m_pads_end; }
                void set_pads_end(const CoordinateDiff& pads_end) { m_pads_end = pads_end; }
                /// \return The auto pad.
                const PadType& get_auto_pad() const { return m_auto_pad; }
                void set_auto_pad(const PadType& auto_pad) { m_auto_pad = auto_pad; }
                /// \return The output padding.
                const CoordinateDiff& get_output_padding() const { return m_output_padding; }
                void set_output_padding(const CoordinateDiff& output_padding)
                {
                    m_output_padding = output_padding;
                }
                bool evaluate(const HostTensorVector &output_values, const HostTensorVector &input_values) override;

            protected:
                Strides m_strides;
                Strides m_dilations;
                CoordinateDiff m_pads_begin;
                CoordinateDiff m_pads_end;
                PadType m_auto_pad;
                CoordinateDiff m_output_padding;
            };
        } // namespace v1

        namespace v0
        {
            /// \brief Group Convolution
            class NGRAPH_API GroupConvolution : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"GroupConvolution", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                GroupConvolution() = default;
                GroupConvolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& window_movement_strides,
                                 const Strides& window_dilation_strides,
                                 const CoordinateDiff& padding_below,
                                 const CoordinateDiff& padding_above,
                                 const Strides& data_dilation_strides,
                                 const size_t groups,
                                 const PadType& pad_type = PadType::EXPLICIT);

                // constructor which accept groups included in filters shape.
                GroupConvolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& window_movement_strides,
                                 const Strides& window_dilation_strides,
                                 const CoordinateDiff& padding_below,
                                 const CoordinateDiff& padding_above,
                                 const Strides& data_dilation_strides,
                                 const PadType& pad_type = PadType::EXPLICIT);
                Shape get_weights_dimensions() const;
                const Strides& get_window_movement_strides() const
                {
                    return m_window_movement_strides;
                }
                const Strides& get_window_dilation_strides() const
                {
                    return m_window_dilation_strides;
                }
                const CoordinateDiff& get_padding_below() const { return m_padding_below; }
                const CoordinateDiff& get_padding_above() const { return m_padding_above; }
                const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
                Output<Node> get_filters() { return input_value(1); }
                Output<Node> get_data_batch() { return input_value(0); }
                size_t get_groups() const { return m_groups; };
                const PadType& get_pad_type() const { return m_pad_type; }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual NodeVector decompose_op() const override;

                virtual void pre_validate_and_infer_types() override;
                virtual void post_validate_and_infer_types() override;

                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

                bool has_groups_in_filters() const { return m_groups_in_filters; }
            protected:
                Strides m_window_movement_strides;
                Strides m_window_dilation_strides;
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                Strides m_data_dilation_strides;
                size_t m_groups;
                PadType m_pad_type{PadType::NOTSET};

            private:
                bool m_groups_in_filters;
            };

            /// \brief Group Convolution data batch backprop
            class NGRAPH_API GroupConvolutionBackpropData : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"GroupConvolutionBackpropData", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                GroupConvolutionBackpropData() = default;
                GroupConvolutionBackpropData(const Output<Node>& data_batch,
                                             const Output<Node>& filters,
                                             const Output<Node>& output_delta,
                                             const Strides& window_movement_strides,
                                             const Strides& window_dilation_strides,
                                             const CoordinateDiff& padding_below,
                                             const CoordinateDiff& padding_above,
                                             const size_t groups);

                const Strides& get_window_movement_strides() const
                {
                    return m_window_movement_strides;
                }
                const Strides& get_window_dilation_strides() const
                {
                    return m_window_dilation_strides;
                }
                const CoordinateDiff& get_padding_below() const { return m_padding_below; }
                const CoordinateDiff& get_padding_above() const { return m_padding_above; }
                size_t get_groups() const { return m_groups; };
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual NodeVector decompose_op() const override;

                virtual void pre_validate_and_infer_types() override;

            protected:
                Strides m_window_movement_strides;
                Strides m_window_dilation_strides;
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                size_t m_groups;
            };
        }

        using v0::GroupConvolution;
        using v0::GroupConvolutionBackpropData;
    } // namespace op
} // namespace ngraph
