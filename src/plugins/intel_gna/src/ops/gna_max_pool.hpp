// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>

#include "ngraph/node.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/max_pool_base.hpp"

namespace ov {
namespace intel_gna {
namespace op {
/// \brief Batched max pooling operation.
class GNAMaxPool : public ov::op::Op {
public:
    OPENVINO_OP("GNAMaxPool", "intel_gna", ov::op::Op);

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
    GNAMaxPool(const ov::Output<ov::Node>& arg,
               const ov::Strides& strides,
               const ov::Shape& pads_begin,
               const ov::Shape& pads_end,
               const ov::Shape& kernel,
               const ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
               const ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT);

    void validate_and_infer_types() override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    /// \return The kernel shape.
    const ov::Shape& get_kernel() const {
        return m_kernel;
    }
    void set_kernel(const ov::Shape& kernel) {
        m_kernel = kernel;
    }
    /// \return The strides.
    const ov::Strides& get_strides() const {
        return m_strides;
    }
    void set_strides(const ov::Strides& strides) {
        m_strides = strides;
    }
    /// \return The beginning of padding shape.
    const ov::Shape& get_pads_begin() const {
        return m_pads_begin;
    }
    void set_pads_begin(const ov::Shape& pads_begin) {
        m_pads_begin = pads_begin;
    }
    /// \return The end of padding shape.
    const ov::Shape& get_pads_end() const {
        return m_pads_end;
    }
    void set_adding_above(const ov::Shape& pads_end) {
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

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

protected:
    bool update_auto_padding(const ov::PartialShape& in_shape,
                             const ov::Strides& filter_dilations,
                             ov::Shape& new_pads_end,
                             ov::Shape& new_pads_begin) const;

    ov::PartialShape infer_output_shape(const ov::Strides& dilations);

    ov::Shape m_kernel;
    ov::Strides m_strides;
    ov::Shape m_pads_begin;
    ov::Shape m_pads_end;
    ov::op::PadType m_auto_pad;
    ov::op::RoundingType m_rounding_type;
};
}  // namespace op
}  // namespace intel_gna
}  // namespace ov