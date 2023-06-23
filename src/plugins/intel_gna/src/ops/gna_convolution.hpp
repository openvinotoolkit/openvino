// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gna {
namespace op {

class GNAConvolution;

namespace internal {

int64_t calculate_num_spatial(const ov::intel_gna::op::GNAConvolution* op,
                              const ov::PartialShape& input_shape,
                              const ov::PartialShape& filters_shape,
                              const int64_t& num_non_spatial_data_dims,
                              const int64_t& num_non_spatial_filter_dims);

void update_and_validate_attributes(ov::intel_gna::op::GNAConvolution* op);

template <class T>
bool resolve_auto_pad_for_shape(const ov::intel_gna::op::GNAConvolution* op,
                                ov::CoordinateDiff& pads_begin,
                                ov::CoordinateDiff& pads_end,
                                const std::vector<T>& input_shapes,
                                const int64_t& num_non_spatial_data_dims,
                                const int64_t& num_non_spatial_filter_dims);
template <class T>
void shape_infer(const ov::intel_gna::op::GNAConvolution* op,
                 const ov::CoordinateDiff& pads_begin,
                 const ov::CoordinateDiff& pads_end,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes);

}  // namespace internal

/**
 * @brief Activation modes for fused convolutions.
 *
 */
enum class ActivationType { SIGMOID, RELU, TANH, ABS, LOG, EXP, SIGN, CLAMP, NO_ACTIVATION };

/// \brief Convolution with NHWC layout
///
class GNAConvolution : public ov::op::Op {
public:
    OPENVINO_OP("GNAConvolution", "intel_gna", ov::op::Op);

    /// \brief Constructs a convolution operation.
    GNAConvolution() = default;
    /// \brief Constructs a convolution operation.
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
    GNAConvolution(const ov::Output<ov::Node>& data_batch,
                   const ov::Output<ov::Node>& filters,
                   const ov::Output<ov::Node>& bias,
                   const ov::Strides& strides,
                   const ov::CoordinateDiff& pads_begin,
                   const ov::CoordinateDiff& pads_end,
                   const ov::Strides& dilations,
                   const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT);

    GNAConvolution(const ov::Output<ov::Node>& data_batch,
                   const ov::Output<ov::Node>& filters,
                   const ov::Strides& strides,
                   const ov::CoordinateDiff& pads_begin,
                   const ov::CoordinateDiff& pads_end,
                   const ov::Strides& dilations,
                   const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT);

    void validate_and_infer_types() override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    /// \return The strides.
    const ov::Strides& get_strides() const {
        return m_strides;
    }
    void set_strides(const ov::Strides& strides) {
        m_strides = strides;
    }
    /// \return The dilations.
    const ov::Strides& get_dilations() const {
        return m_dilations;
    }
    void set_dilations(const ov::Strides& dilations) {
        m_dilations = dilations;
    }
    /// \return The padding-below sizes (possibly negative).
    const ov::CoordinateDiff& get_pads_begin() const {
        return m_pads_begin;
    }
    void set_pads_begin(const ov::CoordinateDiff& pads_begin) {
        m_pads_begin = pads_begin;
    }
    /// \return The padding-above sizes (possibly negative).
    const ov::CoordinateDiff& get_pads_end() const {
        return m_pads_end;
    }
    void set_adding_above(const ov::CoordinateDiff& pads_end) {
        m_pads_end = pads_end;
    }
    /// \return The pad type for pooling.
    ov::op::PadType get_auto_pad() const {
        return m_auto_pad;
    }
    void set_auto_pad(const ov::op::PadType& auto_pad) {
        m_auto_pad = auto_pad;
    }
    bool has_bias() const {
        return m_has_add_node;
    }
    ActivationType get_activation() const {
        return m_activation_type;
    }
    void set_activation(ActivationType activation_type) {
        m_activation_type = activation_type;
    }

protected:
    ov::Strides m_strides;
    ov::Strides m_dilations;
    ov::CoordinateDiff m_pads_begin;
    ov::CoordinateDiff m_pads_end;
    ov::op::PadType m_auto_pad;
    int64_t m_num_spatial = -1;

private:
    friend int64_t internal::calculate_num_spatial(const ov::intel_gna::op::GNAConvolution* op,
                                                   const ov::PartialShape& input_shape,
                                                   const ov::PartialShape& filters_shape,
                                                   const int64_t& num_non_spatial_data_dims,
                                                   const int64_t& num_non_spatial_filter_dims);

    friend void internal::update_and_validate_attributes(ov::intel_gna::op::GNAConvolution* op);

    template <class T>
    friend bool internal::resolve_auto_pad_for_shape(const ov::intel_gna::op::GNAConvolution* op,
                                                     ov::CoordinateDiff& pads_begin,
                                                     ov::CoordinateDiff& pads_end,
                                                     const std::vector<T>& input_shapes,
                                                     const int64_t& num_non_spatial_data_dims,
                                                     const int64_t& num_non_spatial_filter_dims);
    template <class T>
    friend void internal::shape_infer(const ov::intel_gna::op::GNAConvolution* op,
                                      const ov::CoordinateDiff& pads_begin,
                                      const ov::CoordinateDiff& pads_end,
                                      const std::vector<T>& input_shapes,
                                      std::vector<T>& output_shapes);
    bool m_has_add_node;
    ActivationType m_activation_type;
};
}  // namespace op
}  // namespace intel_gna
}  // namespace ov