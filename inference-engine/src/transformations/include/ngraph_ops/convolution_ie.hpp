// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <algorithm>

#include <ie_api.h>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(ConvolutionIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"ConvolutionIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    /// \brief Constructs a batched convolution operation.
    ConvolutionIE() = default;
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
    ConvolutionIE(const Output<Node>& data_batch,
                  const Output<Node>& filters,
                  const Strides& strides,
                  const Strides& dilations,
                  const CoordinateDiff& pads_begin,
                  const CoordinateDiff& pads_end,
                  const size_t& group = 1,
                  const PadType& auto_pad = PadType::EXPLICIT);

    ConvolutionIE(const Output<Node>& data_batch,
                  const Output<Node>& filters,
                  const Output<Node>& bias,
                  const Strides& strides,
                  const Strides& dilations,
                  const CoordinateDiff& pads_begin,
                  const CoordinateDiff& pads_end,
                  const size_t& group = 1,
                  const PadType& auto_pad = PadType::EXPLICIT);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & new_args) const override;

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
    /// \return The groups for convolution.
    const size_t& get_group() const { return m_group; }
    void set_group(const size_t & group) { m_group = group; }

protected:
    Strides m_strides;
    Strides m_dilations;
    CoordinateDiff m_pads_begin;
    CoordinateDiff m_pads_end;
    PadType m_auto_pad;
    size_t m_group;
};

}  // namespace op
}  // namespace ngraph
