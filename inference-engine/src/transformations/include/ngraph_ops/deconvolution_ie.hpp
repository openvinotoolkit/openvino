// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <algorithm>

#include <transformations_visibility.hpp>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API DeconvolutionIE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"DeconvolutionIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    DeconvolutionIE() = default;

    DeconvolutionIE(const Output<Node>& data,
                    const Output<Node>& filters,
                    const Strides& strides,
                    const Strides& dilations,
                    const CoordinateDiff& pads_begin,
                    const CoordinateDiff& pads_end,
                    const element::Type output_type,
                    const size_t& group = 1,
                    const PadType& auto_pad = PadType::EXPLICIT,
                    const CoordinateDiff& output_padding = {},
                    const std::shared_ptr<Node> & output_shape = nullptr);

    DeconvolutionIE(const Output<Node>& data,
                    const Output<Node>& filters,
                    const Output<Node>& bias,
                    const Strides& strides,
                    const Strides& dilations,
                    const CoordinateDiff& pads_begin,
                    const CoordinateDiff& pads_end,
                    const element::Type output_type,
                    const size_t& group = 1,
                    const PadType& auto_pad = PadType::EXPLICIT,
                    const CoordinateDiff& output_padding = {},
                    const std::shared_ptr<Node> & output_shape = nullptr);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & new_args) const override;

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
    /// \return The group
    const size_t& get_group() const { return m_group; }
    void set_group(const size_t & group) { m_group = group; }
    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    Strides m_strides;
    Strides m_dilations;
    CoordinateDiff m_pads_begin;
    CoordinateDiff m_pads_end;
    PadType m_auto_pad;
    size_t m_group;
    CoordinateDiff m_output_padding;
    std::shared_ptr<Node> m_output_shape;
    element::Type m_output_type;
};

}  // namespace op
}  // namespace ngraph
