// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph {
namespace op {
namespace v1 {
/// \brief Batched convolution operation, with optional window dilation and stride.
///
class NGRAPH_API Convolution : public Op {
public:
    NGRAPH_RTTI_DECLARATION;

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

    virtual std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

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
    void set_adding_above(const CoordinateDiff& pads_end) {
        m_pads_end = pads_end;
    }
    /// \return The pad type for convolution.
    const PadType& get_auto_pad() const {
        return m_auto_pad;
    }
    void set_auto_pad(const PadType& auto_pad) {
        m_auto_pad = auto_pad;
    }
    /// \return The default value for Convolution.
    NGRAPH_SUPPRESS_DEPRECATED_START
    virtual std::shared_ptr<Node> get_default_value() const override;
    NGRAPH_SUPPRESS_DEPRECATED_END

    template <class ShapeType>
    void shape_infer(ShapeType& input_shape, ShapeType& filters_shape, ShapeType& output_shape);

protected:
    template <class ShapeType>
    void calculate_num_spatial_dims_and_update_attributes(ShapeType& input_shape,
                                                          ShapeType& filters_shape,
                                                          Strides& dilations,
                                                          Strides& strides,
                                                          CoordinateDiff& pad_begin,
                                                          CoordinateDiff& pad_end,
                                                          const op::PadType& auto_pad);

    Strides m_strides;
    Strides m_dilations;
    CoordinateDiff m_pads_begin;
    CoordinateDiff m_pads_end;
    PadType m_auto_pad;

    bool need_revalidation{true};
    int64_t num_spatials;
};

template <class ShapeType>
void op::v1::Convolution::calculate_num_spatial_dims_and_update_attributes(ShapeType& input_shape,
                                                                           ShapeType& filters_shape,
                                                                           Strides& dilations,
                                                                           Strides& strides,
                                                                           CoordinateDiff& pad_begin,
                                                                           CoordinateDiff& pad_end,
                                                                           const op::PadType& auto_pad) {
    const auto& input_rank = input_shape.rank();
    const auto& filters_rank = filters_shape.rank();
    if (need_revalidation || num_spatials == -1) {
        num_spatials = -1;
        if (const auto& size = dilations.size())
            num_spatials = static_cast<int64_t>(size);
        if (const auto& size = strides.size())
            num_spatials = static_cast<int64_t>(size);
        if (const auto& size = pad_begin.size())
            num_spatials = static_cast<int64_t>(size);
        if (const auto& size = pad_end.size())
            num_spatials = static_cast<int64_t>(size);
        if (input_rank.is_static())
            num_spatials = input_rank.get_length() - 2;
        if (filters_rank.is_static())
            num_spatials = filters_rank.get_length() - 2;

        if (num_spatials == -1)
            return;  // can not deduce output rank

        if (strides.empty()) {
            strides = Strides(num_spatials, 1);
            set_strides(strides);
        }
        if (dilations.empty()) {
            dilations = Strides(num_spatials, 1);
            set_dilations(dilations);
        }
        if (pad_begin.empty() || auto_pad == op::PadType::VALID) {
            pad_begin = CoordinateDiff(num_spatials, 0);
            set_pads_begin(pad_begin);
        }
        if (pad_end.empty() || auto_pad == op::PadType::VALID) {
            pad_end = CoordinateDiff(num_spatials, 0);
            set_adding_above(pad_end);
        }

        NODE_VALIDATION_CHECK(this,
                              strides.size() == num_spatials,
                              "Strides should be defined for all and only spatial features.");
        NODE_VALIDATION_CHECK(this,
                              dilations.size() == num_spatials,
                              "Dilations should be defined for all and only spatial features.");
        NODE_VALIDATION_CHECK(this,
                              pad_begin.size() == num_spatials && pad_end.size() == num_spatials,
                              "Pads should be defined for all and only spatial features.");
        NODE_VALIDATION_CHECK(this,
                              std::all_of(dilations.begin(),
                                          dilations.end(),
                                          [](const size_t& i) {
                                              return i > 0;
                                          }),
                              "Filter dilation (",
                              dilations,
                              ") has zero dimension.");
        NODE_VALIDATION_CHECK(this,
                              std::all_of(strides.begin(),
                                          strides.end(),
                                          [](const size_t& i) {
                                              return i > 0;
                                          }),
                              "Filter strides (",
                              strides,
                              ") has zero dimension.");
        need_revalidation = false;
    }

    if (input_rank.is_dynamic())
        input_shape.resize(num_spatials + 2);
    if (filters_rank.is_dynamic())
        filters_shape.resize(num_spatials + 2);

    NODE_VALIDATION_CHECK(this,
                          (input_shape.size() == (num_spatials + 2)) && (filters_shape.size() == (num_spatials + 2)),
                          "Data batch and filters rank do not match (data batch shape: ",
                          input_shape,
                          ", filters shape: ",
                          filters_shape,
                          ").");
}

template <class ShapeType>
void op::v1::Convolution::shape_infer(ShapeType& input_shape, ShapeType& filters_shape, ShapeType& output_shape) {
    auto dilations = get_dilations();
    auto strides = get_strides();
    auto pad_begin = get_pads_begin(), pad_end = get_pads_end();
    const auto& auto_pad = get_auto_pad();

    calculate_num_spatial_dims_and_update_attributes(input_shape,
                                                     filters_shape,
                                                     dilations,
                                                     strides,
                                                     pad_begin,
                                                     pad_end,
                                                     auto_pad);
    if (num_spatials < 1)
        return;
    // ranks are originally static or aligned with num_spatials, attributes are valid

    output_shape.resize(num_spatials + 2);
    output_shape[0] = input_shape[0];
    output_shape[1] = filters_shape[0];

    NODE_VALIDATION_CHECK(
        this,
        input_shape[1].is_dynamic() || filters_shape[1].is_dynamic() || input_shape[1] == filters_shape[1],
        "Data batch channel count (",
        input_shape[1],
        ") does not match filter input ",
        "channel count (",
        filters_shape[1],
        ").");

    for (size_t i = 0; i < num_spatials; ++i) {
        const auto& input_dim = input_shape[i + 2];
        const auto& filters_dim = filters_shape[i + 2];
        if (input_dim.is_static() && filters_dim.is_static()) {
            const int64_t& window_dilated_dim = (filters_dim.get_length() - 1) * dilations[i] + 1;
            NODE_VALIDATION_CHECK(this,
                                  window_dilated_dim > 0,
                                  "Window after dilation has dimension less than 1 (dim: ",
                                  window_dilated_dim,
                                  ") at axis ",
                                  i,
                                  ".");
            if (auto_pad == op::PadType::SAME_UPPER || auto_pad == op::PadType::SAME_LOWER) {
                const int64_t& image_size = input_dim.get_length();
                const int64_t& filter_stride = strides[i];
                const int64_t& output_size = (image_size + filter_stride - 1) / filter_stride;

                const int64_t& tmp = (output_size - 1) * filter_stride + window_dilated_dim;
                const int64_t& padding_needed = tmp > image_size ? tmp - image_size : 0;

                const size_t& padding_lhs = static_cast<size_t>(padding_needed / 2);
                const size_t& padding_rhs = static_cast<size_t>(padding_needed - padding_lhs);

                pad_begin[i] = auto_pad == op::PadType::SAME_UPPER ? padding_lhs : padding_rhs;
                pad_end[i] = auto_pad == op::PadType::SAME_UPPER ? padding_rhs : padding_lhs;
            }

            const int64_t& data_padded_dilated_dim = input_dim.get_length() + pad_begin[i] + pad_end[i];
            NODE_VALIDATION_CHECK(this,
                                  window_dilated_dim <= data_padded_dilated_dim,
                                  "Window after dilation has dimension (dim: ",
                                  window_dilated_dim,
                                  ") larger than the data shape after padding (dim: ",
                                  data_padded_dilated_dim,
                                  ") at axis ",
                                  i,
                                  ".");
            output_shape[i + 2] = (data_padded_dilated_dim - window_dilated_dim) / strides[i] + 1;
        }
    }
    set_pads_begin(pad_begin);
    set_adding_above(pad_end);
}

/// \brief Data batch backprop for batched convolution operation.
class NGRAPH_API ConvolutionBackpropData : public Op {
public:
    NGRAPH_RTTI_DECLARATION;

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
    } // namespace op
} // namespace ngraph
