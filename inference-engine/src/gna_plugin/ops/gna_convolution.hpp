// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "ngraph/node.hpp"
#include <transformations_visibility.hpp>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"

namespace GNAPluginNS {
namespace Op {

class GNAConvolution;

namespace internal {

int64_t calculate_num_spatial(const GNAPluginNS::Op::GNAConvolution * op,
                                         const ngraph::PartialShape& input_shape,
                                         const ngraph::PartialShape& filters_shape,
                                         const int64_t& num_non_spatial_data_dims,
                                         const int64_t& num_non_spatial_filter_dims);

void update_and_validate_attributes(GNAPluginNS::Op::GNAConvolution* op);

template <class T>
bool resolve_auto_pad_for_shape(const GNAPluginNS::Op::GNAConvolution* op,
                                           ngraph::CoordinateDiff& pads_begin,
                                           ngraph::CoordinateDiff& pads_end,
                                           const std::vector<T>& input_shapes,
                                           const int64_t& num_non_spatial_data_dims,
                                           const int64_t& num_non_spatial_filter_dims);
template <class T>
void shape_infer(const GNAPluginNS::Op::GNAConvolution* op,
                            const ngraph::CoordinateDiff& pads_begin,
                            const ngraph::CoordinateDiff& pads_end,
                            const std::vector<T>& input_shapes,
                            std::vector<T>& output_shapes);

} // namespace internal

/// \brief Convolution with NHWC layout
///
class GNAConvolution : public ov::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;

    /// \brief Constructs a batched convolution operation.
    GNAConvolution() = default;
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
    GNAConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                const ngraph::Output<ngraph::Node>& filters,
                const ngraph::Output<ngraph::Node>& bias,
                const ngraph::Strides& strides,
                const ngraph::CoordinateDiff& pads_begin,
                const ngraph::CoordinateDiff& pads_end,
                const ngraph::Strides& dilations,
                const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT);

    GNAConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                const ngraph::Output<ngraph::Node>& filters,
                const ngraph::Strides& strides,
                const ngraph::CoordinateDiff& pads_begin,
                const ngraph::CoordinateDiff& pads_end,
                const ngraph::Strides& dilations,
                const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT);

    void validate_and_infer_types() override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    /// \return The strides.
    const ngraph::Strides& get_strides() const {
        return m_strides;
    }
    void set_strides(const ngraph::Strides& strides) {
        m_strides = strides;
    }
    /// \return The dilations.
    const ngraph::Strides& get_dilations() const {
        return m_dilations;
    }
    void set_dilations(const ngraph::Strides& dilations) {
        m_dilations = dilations;
    }
    /// \return The padding-below sizes (possibly negative).
    const ngraph::CoordinateDiff& get_pads_begin() const {
        return m_pads_begin;
    }
    void set_pads_begin(const ngraph::CoordinateDiff& pads_begin) {
        m_pads_begin = pads_begin;
    }
    /// \return The padding-above sizes (possibly negative).
    const ngraph::CoordinateDiff& get_pads_end() const {
        return m_pads_end;
    }
    void set_adding_above(const ngraph::CoordinateDiff& pads_end) {
        m_pads_end = pads_end;
    }
    /// \return The pad type for convolution.
    const ov::op::PadType& get_auto_pad() const {
        return m_auto_pad;
    }
    void set_auto_pad(const ov::op::PadType& auto_pad) {
        m_auto_pad = auto_pad;
    }

    /*
     * TODO: for unit tests
    bool evaluate(ov::runtime::TensorVector& output_values,
                  const ov::runtime::TensorVector& input_values,
                  const ov::EvaluationContext & evaluation_context) const override;
    bool has_evaluate() const override;
    */

protected:
    ngraph::Strides m_strides;
    ngraph::Strides m_dilations;
    ngraph::CoordinateDiff m_pads_begin;
    ngraph::CoordinateDiff m_pads_end;
    ov::op::PadType m_auto_pad;
    int64_t m_num_spatial = -1;

private:
    friend int64_t internal::calculate_num_spatial(const GNAPluginNS::Op::GNAConvolution* op,
                                         const ngraph::PartialShape& input_shape,
                                         const ngraph::PartialShape& filters_shape,
                                         const int64_t& num_non_spatial_data_dims,
                                         const int64_t& num_non_spatial_filter_dims);

    friend void internal::update_and_validate_attributes(GNAPluginNS::Op::GNAConvolution* op);

    template <class T>
    friend bool internal::resolve_auto_pad_for_shape(const GNAPluginNS::Op::GNAConvolution* op,
                                           ngraph::CoordinateDiff& pads_begin,
                                           ngraph::CoordinateDiff& pads_end,
                                           const std::vector<T>& input_shapes,
                                           const int64_t& num_non_spatial_data_dims,
                                           const int64_t& num_non_spatial_filter_dims);
    template <class T>
    friend void internal::shape_infer(const GNAPluginNS::Op::GNAConvolution* op,
                            const ngraph::CoordinateDiff& pads_begin,
                            const ngraph::CoordinateDiff& pads_end,
                            const std::vector<T>& input_shapes,
                            std::vector<T>& output_shapes);
};
} // namespace Op
}  // namespace GNAPluginNS
