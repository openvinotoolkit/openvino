// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

/// \brief Operator performing Gated Linear Unit Activation
/// This operation performs gated linear unit activation that combines swish or gelu activation function
class TRANSFORMATIONS_API GLU : public ov::op::Op {
public:
    OPENVINO_OP("GLU", "ie_internal_opset");

    enum GluType { Swish = 0, Gelu, Gelu_Tanh };

    GLU() = default;
    /// \brief Constructs an GLU operation.
    ///
    /// \param data Input tensor with data
    /// \param axis The index of an axis in "data" along which to perform the split
    /// \param split_lenghts A list containing the sizes of each output tensor along the split "axis"
    /// \param glu_type GLU type, one of Swish, Gelu and Gelu_Tanh
    /// \param split_to_glu_idx Output index of variadic split, which is connected to GLU
    /// \param output_type Output element type
    GLU(const Output<Node>& data,
        int64_t axis,
        int64_t split_lengths,
        const GluType glu_type,
        const size_t split_to_glu_idx,
        const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    int64_t get_axis() const {
        return m_axis;
    }
    int64_t get_split_lengths() const {
        return m_split_lengths;
    }
    GluType get_glu_type() const {
        return m_glu_type;
    }
    size_t get_split_to_glu_idx() const {
        return m_split_to_glu_idx;
    }

    void set_axis(int64_t axis) {
        m_axis = axis;
    }
    void set_split_lengths(int64_t split_lengths) {
        m_split_lengths = split_lengths;
    }
    void set_glu_type(GluType glu_type) {
        m_glu_type = glu_type;
    }
    void set_split_to_glu_idx(size_t split_to_glu_idx) {
        m_split_to_glu_idx = split_to_glu_idx;
    }

private:
    int64_t m_axis = 0;
    int64_t m_split_lengths = 0;
    GluType m_glu_type = GluType::Swish;
    size_t m_split_to_glu_idx = 0;
    ov::element::Type m_output_type{};
};

}  // namespace internal
}  // namespace op

std::ostream& operator<<(std::ostream& s, const op::internal::GLU::GluType& reduction);

template <>
class AttributeAdapter<op::internal::GLU::GluType> : public EnumAttributeAdapterBase<op::internal::GLU::GluType> {
public:
    AttributeAdapter(op::internal::GLU::GluType& value) : EnumAttributeAdapterBase<op::internal::GLU::GluType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<op::internal::GLU::GluType>");
};

}  // namespace ov
