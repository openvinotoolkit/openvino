// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/convolution.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Batched convolution operation, with optional window dilation and stride.
class OPENVINO_API GroupConvolution : public Op {
public:
    OPENVINO_OP("GroupConvolution", "opset1", op::Op, 1);
    BWDCMP_RTTI_DECLARATION;

    /// \brief Constructs a batched convolution operation.
    GroupConvolution() = default;
    /// \brief Constructs a batched convolution operation.
    ///
    /// \param data_batch The node producing the input data batch tensor.<br>
    /// `[N, C_IN, D1, ... Df]`
    /// \param filters The node producing the filters tensor.<br>
    /// `[GROUPS, FC_OUT, FC_IN, F1, ... Ff]`
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
    /// Output `[N, FC_OUT * GROUPS, R1, ... Rf]`
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

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    /// \return The strides.
    const Strides& get_strides() const {
        return m_strides;
    }
    void set_strides(const Strides& strides) {
        m_strides = strides;
    }
    /// \return The dilations.
    const Strides& get_dilations() const {
        return m_dilations;
    }
    void set_dilations(const Strides& dilations) {
        m_dilations = dilations;
    }
    /// \return The padding-below sizes (possibly negative).
    const CoordinateDiff& get_pads_begin() const {
        return m_pads_begin;
    }
    void set_pads_begin(const CoordinateDiff& pads_begin) {
        m_pads_begin = pads_begin;
    }
    /// \return The padding-above sizes (possibly negative).
    const CoordinateDiff& get_pads_end() const {
        return m_pads_end;
    }
    void set_pads_end(const CoordinateDiff& pads_end) {
        m_pads_end = pads_end;
    }
    OPENVINO_DEPRECATED("This method is deprecated and will be removed soon. Please use set_pads_end instead.")
    void set_adding_above(const CoordinateDiff& pads_end) {
        set_pads_end(pads_end);
    }
    /// \return The pad type for convolution.
    const PadType& get_auto_pad() const {
        return m_auto_pad;
    }
    void set_auto_pad(const PadType& auto_pad) {
        m_auto_pad = auto_pad;
    }
    /// \return The default value for Convolution.
    OPENVINO_SUPPRESS_DEPRECATED_START
    std::shared_ptr<Node> get_default_value() const override;
    OPENVINO_SUPPRESS_DEPRECATED_END

protected:
    Strides m_strides;
    Strides m_dilations;
    CoordinateDiff m_pads_begin;
    CoordinateDiff m_pads_end;
    PadType m_auto_pad;
    int64_t m_num_spatial = -1;

private:
    template <class ConvType>
    friend int64_t calculate_num_spatial(const ConvType* op,
                                         const PartialShape& input_shape,
                                         const PartialShape& filters_shape,
                                         const int64_t& num_non_spatial_data_dims,
                                         const int64_t& num_non_spatial_filter_dims);

    template <class ConvType>
    friend void update_and_validate_attributes(ConvType* op, int64_t num_spatial);
    template <class ConvType, class ShapeType>
    friend bool resolve_auto_pad_for_shape(const ConvType* op,
                                           CoordinateDiff& pads_begin,
                                           CoordinateDiff& pads_end,
                                           const std::vector<ShapeType>& input_shapes,
                                           const int64_t& num_non_spatial_data_dims,
                                           const int64_t& num_non_spatial_filter_dims);

    template <class T>
    friend void shape_infer(const GroupConvolution* op,
                            const CoordinateDiff& pads_begin,
                            const CoordinateDiff& pads_end,
                            const std::vector<T>& input_shapes,
                            std::vector<T>& output_shapes);
};

/// \brief Data batch backprop for batched convolution operation.
class OPENVINO_API GroupConvolutionBackpropData : public Op {
public:
    OPENVINO_OP("GroupConvolutionBackpropData", "opset1", op::Op, 1);
    BWDCMP_RTTI_DECLARATION;

    /// \brief Constructs a batched-convolution data batch-backprop operation.
    GroupConvolutionBackpropData();
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
    void infer_conv_backprop_output_spatial_shape(const std::vector<Dimension>& input_data_shape,
                                                  const std::vector<Dimension>& filters_shape,
                                                  const Strides& strides,
                                                  const Strides& dilations,
                                                  const CoordinateDiff& pads_begin,
                                                  const CoordinateDiff& pads_end,
                                                  const CoordinateDiff& output_padding,
                                                  std::vector<Dimension>& output_spatial_shape);

    bool visit_attributes(AttributeVisitor& visitor) override;
    bool is_dynamic() const override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return The spatial shape of the output.
    const PartialShape get_convolution_output_shape() const;
    void set_output_shape(const Shape& output_shape);
    /// \return The strides from the forward prop.
    const Strides& get_strides() const {
        return m_strides;
    }
    void set_strides(const Strides& strides) {
        m_strides = strides;
    }
    /// \return The dilations from the forward prop.
    const Strides& get_dilations() const {
        return m_dilations;
    }
    void set_dilations(const Strides& dilations) {
        m_dilations = dilations;
    }
    /// \return The number of pixels to add to the beginning along each axis.
    const CoordinateDiff& get_pads_begin() const {
        return m_pads_begin;
    }
    void set_pads_begin(const CoordinateDiff& pads_begin) {
        m_pads_begin = pads_begin;
    }
    /// \return The number of pixels to add to the ending along each axis.
    const CoordinateDiff& get_pads_end() const {
        return m_pads_end;
    }
    void set_pads_end(const CoordinateDiff& pads_end) {
        m_pads_end = pads_end;
    }
    /// \return The auto pad.
    const PadType& get_auto_pad() const {
        return m_auto_pad;
    }
    void set_auto_pad(const PadType& auto_pad) {
        m_auto_pad = auto_pad;
    }
    /// \return The output padding.
    const CoordinateDiff& get_output_padding() const {
        return m_output_padding;
    }
    void set_output_padding(const CoordinateDiff& output_padding) {
        m_output_padding = output_padding;
    }

protected:
    Strides m_strides;
    Strides m_dilations;
    CoordinateDiff m_pads_begin;
    CoordinateDiff m_pads_end;
    PadType m_auto_pad;
    CoordinateDiff m_output_padding;

    int64_t m_num_spatial = -1;

private:
    template <class ConvType>
    friend int64_t calculate_num_spatial(const ConvType* op,
                                         const PartialShape& input_shape,
                                         const PartialShape& filters_shape,
                                         const PartialShape& output_shapes_shape,
                                         const int64_t& num_non_spatial_data_dims,
                                         const int64_t& num_non_spatial_filter_dims);

    template <class ConvType>
    friend int64_t calculate_num_spatial(const ConvType* op,
                                         const PartialShape& input_shape,
                                         const PartialShape& filters_shape,
                                         const int64_t& num_non_spatial_data_dims,
                                         const int64_t& num_non_spatial_filter_dims);

    template <class ConvType>
    friend void update_and_validate_attributes(ConvType* op, int64_t num_spatial);

    template <class ConvType>
    friend void update_and_validate_attributes_back_prop(ConvType* op, int64_t num_spatial);

    template <class ConvType, class ShapeType>
    friend bool resolve_auto_pad_for_shape_back_prop(const ConvType* op,
                                                     CoordinateDiff& pads_begin,
                                                     CoordinateDiff& pads_end,
                                                     const std::vector<ShapeType>& input_shapes,
                                                     ShapeType& output_spatial_shape,
                                                     const int64_t& num_non_spatial_data_dims,
                                                     const int64_t& num_non_spatial_filter_dims);

    template <class T>
    friend void shape_infer(const GroupConvolutionBackpropData* op,
                            const CoordinateDiff& pads_begin,
                            const CoordinateDiff& pads_end,
                            const T& output_shape_from_input,
                            const std::vector<T>& input_shapes,
                            std::vector<T>& output_shapes);
};
}  // namespace v1
}  // namespace op
}  // namespace ov
