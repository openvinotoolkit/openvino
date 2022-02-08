// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "ngraph/node.hpp"
#include "openvino/op/util/max_pool_base.hpp"

namespace GNAPluginNS {
namespace Op {
/// \brief Batched max pooling operation.
class GNAMaxPool : public ov::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;

    /// \brief Constructs a batched max pooling operation.
    GNAMaxPool() = default;

    /// \brief Constructs a batched max pooling operation.
    ///
    /// \param arg The node producing the input data batch tensor.
    /// \param strides The strides.
    /// \param pads_begin The beginning of padding shape.
    /// \param pads_end The end of padding shape.
    /// \param kernel The kernel shape.
    /// \param rounding_type Whether to use ceiling or floor rounding type while
    /// computing output shape.
    /// \param auto_pad The pad type for automatically computing padding sizes.
    GNAMaxPool(const ngraph::Output<ngraph::Node>& arg,
            const ngraph::Strides& strides,
            const ngraph::Shape& pads_begin,
            const ngraph::Shape& pads_end,
            const ngraph::Shape& kernel,
            const ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
            const ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT);

    void validate_and_infer_types() override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    /// \return The kernel shape.
    const ngraph::Shape& get_kernel() const {
        return m_kernel;
    }
    void set_kernel(const ngraph::Shape& kernel) {
        m_kernel = kernel;
    }
    /// \return The strides.
    const ngraph::Strides& get_strides() const {
        return m_strides;
    }
    void set_strides(const ngraph::Strides& strides) {
        m_strides = strides;
    }
    /// \return The beginning of padding shape.
    const ngraph::Shape& get_pads_begin() const {
        return m_pads_begin;
    }
    void set_pads_begin(const ngraph::Shape& pads_begin) {
        m_pads_begin = pads_begin;
    }
    /// \return The end of padding shape.
    const ngraph::Shape& get_pads_end() const {
        return m_pads_end;
    }
    void set_adding_above(const ngraph::Shape& pads_end) {
        m_pads_end = pads_end;
    }
    /// \return The pad type for pooling.
    ov::op::PadType get_auto_pad() const {
        return m_auto_pad;
    }
    void set_auto_pad(const ov::op::PadType auto_pad) {
        m_auto_pad = auto_pad;
    }
    /// \return The ceiling mode being used for output shape computations
    ov::op::RoundingType get_rounding_type() const {
        return m_rounding_type;
    }
    void set_rounding_type(ov::op::RoundingType rounding_type) {
        m_rounding_type = rounding_type;
    }

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

protected:
    bool update_auto_padding(const ov::PartialShape& in_shape,
                             const ngraph::Strides& filter_dilations,
                             ngraph::Shape& new_pads_end,
                             ngraph::Shape& new_pads_begin) const;

    ov::PartialShape infer_output_shape(const ngraph::Strides& dilations);

    ngraph::Shape m_kernel;
    ngraph::Strides m_strides;
    ngraph::Shape m_pads_begin;
    ngraph::Shape m_pads_end;
    ov::op::PadType m_auto_pad;
    ov::op::RoundingType m_rounding_type;
};
} // namespace Op
} // namespace GNAPluginNS
