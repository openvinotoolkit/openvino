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
        namespace v1
        {
            /// \brief Batched convolution operation, with optional window dilation and stride.
            ///
            class NGRAPH_API Convolution : public Op
            {
            public:
                RTTI_DECLARATION

                /// \brief Constructs a batched convolution operation.
                Convolution() = default;
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
                Convolution(const Output<Node>& data_batch,
                            const Output<Node>& filters,
                            const Strides& strides,
                            const CoordinateDiff& pads_begin,
                            const CoordinateDiff& pads_end,
                            const Strides& dilations,
                            const PadType& auto_pad = PadType::EXPLICIT);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

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

            protected:
                Strides m_strides;
                Strides m_dilations;
                CoordinateDiff m_pads_begin;
                CoordinateDiff m_pads_end;
                PadType m_auto_pad;
            };

            /// \brief Data batch backprop for batched convolution operation.
            class NGRAPH_API ConvolutionBackpropData : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ConvolutionBackpropData", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched-convolution data batch-backprop operation.
                ConvolutionBackpropData() = default;
                // clang-format off
                //
                // \brief      Constructs a batched-convolution data batch-backprop operation.
                //
                // \param      data            The node producing data from forward-prop. Shape: [N,
                //                             C_INPUT, X1, ..., XD].
                // \param      filters         The node producing the filter from forward-prop. Shape:
                //                             [C_INPUT, C_OUTPUT, K_D, ..., K_1]
                // \param      output_shape    The shape of the data batch from forward-prop. It's size
                //                             should be equal to number of data spatial dimensions.
                // \param      strides         The strides from forward-prop.
                // \param      pads_begin      The padding-below sizes from forward-prop.
                // \param      pads_end        The padding-above sizes from forward-prop.
                // \param      dilations       The dilations from forward-prop.
                // \param      auto_pad        The pad type for automatically computing padding sizes.
                // \param      output_padding  The output padding adds additional amount of paddings per
                //                             each spatial axis in the output tensor. clang-format on
                //
                ConvolutionBackpropData(const Output<Node>& data,
                                        const Output<Node>& filters,
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
                //                             C_INPUT, X1, ..., XD].
                // \param      filters         The node producing the filter from forward-prop. Shape:
                //                             [C_INPUT, C_OUTPUT, K_D, ..., K_1]
                // \param      strides         The strides from forward-prop.
                // \param      pads_begin      The padding-below sizes from forward-prop.
                // \param      pads_end        The padding-above sizes from forward-prop.
                // \param      dilations       The dilations from forward-prop.
                // \param      auto_pad        The pad type for automatically computing padding sizes.
                // \param      output_padding  The output padding adds additional amount of paddings per
                //                             each spatial axis in the output tensor. clang-format on
                //
                ConvolutionBackpropData(const Output<Node>& data,
                                        const Output<Node>& filters,
                                        const Strides& strides,
                                        const CoordinateDiff& pads_begin,
                                        const CoordinateDiff& pads_end,
                                        const Strides& dilations,
                                        const PadType& auto_pad = PadType::EXPLICIT,
                                        const CoordinateDiff& output_padding = {});

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual bool is_dynamic() const override;

                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const OutputVector& deltas) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The output spatial dimensions shape.
                const PartialShape get_output_shape() const;
                void set_output_shape(const Shape& output_shape);
                /// \return The strides from the forward prop.
                const Strides& get_strides() const { return m_strides; }
                void set_strides(const Strides& strides) { m_strides = strides; }
                /// \return The dilations from the forward prop.
                const Strides& get_dilations() const { return m_dilations; }
                void set_dilations(const Strides& dilations) { m_dilations = dilations; }
                /// \return The padding-below sizes (possibly negative) from the forward prop.
                const CoordinateDiff& get_pads_begin() const { return m_pads_begin; }
                void set_pads_begin(const CoordinateDiff& pads_begin) { m_pads_begin = pads_begin; }
                /// \return The padding-above sizes (possibly negative) from the forward prop.
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
                /// \brief      Calculates output spatial features size.
                ///
                /// \param[in]  input_data_shape      The input data partial shape
                /// \param[in]  filters_shape         The filters partial shape
                /// \param[in]  strides               The strides values.
                /// \param[in]  dilations             The dilations values.
                /// \param[in]  pads_begin            The paddings at the beginning of axis.
                /// \param[in]  pads_end              The paddings at the end of axis.
                /// \param[in]  output_padding    The output padding values.
                /// \param      output_spatial_shape  The placeholder for computed output spatial partial
                /// shape.
                ///
                void
                    infer_conv_backprop_output_spatial_shape(const std::vector<Dimension>& input_data_shape,
                                                            const std::vector<Dimension>& filters_shape,
                                                            const Strides& strides,
                                                            const Strides& dilations,
                                                            const CoordinateDiff& pads_begin,
                                                            const CoordinateDiff& pads_end,
                                                            const CoordinateDiff& output_padding,
                                                            std::vector<Dimension>& output_spatial_shape);

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
            /// \brief Batched convolution operation, with optional window dilation and stride.
            ///
            class NGRAPH_API Convolution : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Convolution", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched convolution operation.
                Convolution() = default;
                /// \brief Constructs a batched convolution operation.
                ///
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[f]`
                /// \param window_dilation_strides The window dilation strides.<br>
                /// `[f]`
                /// \param padding_below The padding-below sizes.<br>
                /// `[f]`
                /// \param padding_above The padding-above sizes.<br>
                /// `[f]`
                /// \param data_dilation_strides The data dilation strides.<br>
                /// `[f]`
                /// \param pad_type The pad type for automatically computing padding sizes.<br>
                /// `[f]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                Convolution(const Output<Node>& data_batch,
                            const Output<Node>& filters,
                            const Strides& window_movement_strides,
                            const Strides& window_dilation_strides,
                            const CoordinateDiff& padding_below,
                            const CoordinateDiff& padding_above,
                            const Strides& data_dilation_strides,
                            const PadType& pad_type = PadType::EXPLICIT);

                /// \brief Constructs a batched convolution operation with no data dilation (i.e.,
                /// all
                ///        data dilation strides are 1).
                ///
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[f]`
                /// \param window_dilation_strides The window dilation strides.<br>
                /// `[f]`
                /// \param padding_below The padding-below sizes.<br>
                /// `[f]`
                /// \param padding_above The padding-above sizes.<br>
                /// `[f]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                Convolution(const Output<Node>& data_batch,
                            const Output<Node>& filters,
                            const Strides& window_movement_strides,
                            const Strides& window_dilation_strides,
                            const CoordinateDiff& padding_below,
                            const CoordinateDiff& padding_above);

                /// \brief Constructs a batched convolution operation with no padding or data
                /// dilation
                ///        (i.e., padding above and below are 0 everywhere, and all data dilation
                ///        strides are 1).
                ///
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[f]`
                /// \param window_dilation_strides The window dilation strides.<br>
                /// `[f]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                Convolution(const Output<Node>& data_batch,
                            const Output<Node>& filters,
                            const Strides& window_movement_strides,
                            const Strides& window_dilation_strides);

                /// \brief Constructs a batched convolution operation with no window dilation,
                /// padding,
                ///        or data dilation (i.e., padding above and below are 0 everywhere, and all
                ///        window/data dilation strides are 1).
                ///
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[f]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                Convolution(const Output<Node>& data_batch,
                            const Output<Node>& filters,
                            const Strides& window_movement_strides);

                /// \brief Constructs a batched convolution operation with no window dilation or
                ///        movement stride (i.e., padding above and below are 0 everywhere, and all
                ///        window/data dilation strides and window movement strides are 1).
                ///
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                Convolution(const Output<Node>& data_batch, const Output<Node>& filters);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const OutputVector& deltas) override;

                /// \return The window movement strides.
                const Strides& get_window_movement_strides() const
                {
                    return m_window_movement_strides;
                }
                void set_window_movement_strides(const Strides& window_movement_strides)
                {
                    m_window_movement_strides = window_movement_strides;
                }
                /// \return The window dilation strides.
                const Strides& get_window_dilation_strides() const
                {
                    return m_window_dilation_strides;
                }
                void set_window_dilation_strides(const Strides& window_dilation_strides)
                {
                    m_window_dilation_strides = window_dilation_strides;
                }
                /// \return The padding-below sizes (possibly negative).
                const CoordinateDiff& get_padding_below() const { return m_padding_below; }
                void set_padding_below(const CoordinateDiff& padding_below)
                {
                    m_padding_below = padding_below;
                }
                /// \return The padding-above sizes (possibly negative).
                const CoordinateDiff& get_padding_above() const { return m_padding_above; }
                void set_adding_above(const CoordinateDiff& padding_above)
                {
                    m_padding_above = padding_above;
                }
                /// \return The input data dilation strides.
                const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
                void set_data_dilation_strides(const Strides& data_dilation_strides)
                {
                    m_data_dilation_strides = data_dilation_strides;
                }
                /// \return The pad type for convolution.
                const PadType& get_pad_type() const { return m_pad_type; }
                void set_pad_type(const PadType& pad_type) { m_pad_type = pad_type; }
                /// \return The default value for Convolution.
                virtual std::shared_ptr<Node> get_default_value() const override;

            protected:
                Strides m_window_movement_strides;
                Strides m_window_dilation_strides;
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                Strides m_data_dilation_strides;
                PadType m_pad_type;
            };

            /// \brief Data batch backprop for batched convolution operation.
            class NGRAPH_API ConvolutionBackpropData : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ConvolutionBackpropData", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched-convolution data batch-backprop operation.
                ConvolutionBackpropData() = default;
                ///
                /// \brief      Constructs a batched-convolution data batch-backprop operation.
                ///
                /// \param      data_batch_shape                 The shape of the data batch from
                ///                                              forward-prop.
                /// \param      filters                          The node producing the filters from
                ///                                              forward-prop.
                /// \param      data                             The node producing output delta.
                /// \param      window_movement_strides_forward  The window movement strides from
                ///                                              forward-prop.
                /// \param      window_dilation_strides_forward  The window dilation strides from
                ///                                              forward-prop.
                /// \param      padding_below_forward            The padding-below sizes from
                ///                                              forward-prop.
                /// \param      padding_above_forward            The padding-above sizes from
                ///                                              forward-prop.
                /// \param      data_dilation_strides_forward    The data dilation strides from
                ///                                              forward-prop.
                ///
                ConvolutionBackpropData(const Shape& data_batch_shape,
                                        const Output<Node>& filters,
                                        const Output<Node>& data,
                                        const Strides& window_movement_strides_forward,
                                        const Strides& window_dilation_strides_forward,
                                        const CoordinateDiff& padding_below_forward,
                                        const CoordinateDiff& padding_above_forward,
                                        const Strides& data_dilation_strides_forward);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const OutputVector& deltas) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The data batch shape.
                const Shape& get_data_batch_shape() const { return m_data_batch_shape; }
                void set_data_batch_shape(const Shape& data_batch_shape)
                {
                    m_data_batch_shape = data_batch_shape;
                }
                /// \return The window movement strides from the forward prop.
                const Strides& get_window_movement_strides_forward() const
                {
                    return m_window_movement_strides_forward;
                }
                void set_window_movement_strides_forward(
                    const Strides& window_movement_strides_forward)
                {
                    m_window_movement_strides_forward = window_movement_strides_forward;
                }
                /// \return The window dilation strides from the forward prop.
                const Strides& get_window_dilation_strides_forward() const
                {
                    return m_window_dilation_strides_forward;
                }
                void set_window_dilation_strides_forward(
                    const Strides& window_dilation_strides_forward)
                {
                    m_window_dilation_strides_forward = window_dilation_strides_forward;
                }
                /// \return The padding-below sizes (possibly negative) from the forward prop.
                const CoordinateDiff& get_padding_below_forward() const
                {
                    return m_padding_below_forward;
                }
                void set_padding_below_forward(const CoordinateDiff& padding_below_forward)
                {
                    m_padding_below_forward = padding_below_forward;
                }
                /// \return The padding-above sizes (possibly negative) from the forward prop.
                const CoordinateDiff& get_padding_above_forward() const
                {
                    return m_padding_above_forward;
                }
                void set_padding_above_forward(const CoordinateDiff& padding_above_forward)
                {
                    m_padding_above_forward = padding_above_forward;
                }
                /// \return The input data dilation strides from the forward prop.
                const Strides& get_data_dilation_strides_forward() const
                {
                    return m_data_dilation_strides_forward;
                }
                void set_data_dilation_strides_forward(const Strides& data_dilation_strides_forward)
                {
                    m_data_dilation_strides_forward = data_dilation_strides_forward;
                }

                // Compute the pad_above values to be used if in a convolution
                CoordinateDiff compute_backward_delta_out_pad_above() const;
                CoordinateDiff compute_backward_delta_out_pad_below() const;

            protected:
                Shape m_data_batch_shape;
                Strides m_window_movement_strides_forward;
                Strides m_window_dilation_strides_forward;
                CoordinateDiff m_padding_below_forward;
                CoordinateDiff m_padding_above_forward;
                Strides m_data_dilation_strides_forward;
            };
        } // namespace v0

        namespace util
        {
            // This is a legacy function, retained because the CPU backend uses it for now.
            // TODO: Update CPU backend to use the new stuff in validation_util.hpp, and remove this
            // function.
            NGRAPH_API
            Shape infer_convolution_output_shape(const Node* node,
                                                 const Shape& data_batch_shape,
                                                 const Shape& filters_shape,
                                                 const Strides& window_movement_strides,
                                                 const Strides& window_dilation_strides,
                                                 const CoordinateDiff& padding_below,
                                                 const CoordinateDiff& padding_above,
                                                 const Strides& data_dilation_strides,
                                                 size_t batch_axis_data,
                                                 size_t input_channel_axis_data,
                                                 size_t input_channel_axis_filters,
                                                 size_t output_channel_axis_filters,
                                                 size_t batch_axis_result,
                                                 size_t output_channel_axis_result);
        } // namespace util

        using v0::Convolution;
        using v0::ConvolutionBackpropData;
    } // namespace op
} // namespace ngraph
