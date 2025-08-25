// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief OneHot operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API OneHot : public Op {
public:
    OPENVINO_OP("OneHot", "opset1", op::Op);

    /// \brief Lists the supported negative indices modes for this version of the operator.
    ///        See the specification for the description of how negative indices are handled.
    enum class NegativeIndicesMode { IGNORE_NEGATIVE, NORMALIZE };

    /// \brief Constructs a one-hot operation.
    OneHot() = default;
    /// \brief Constructs a one-hot operation.
    ///
    /// \param indices   Input tensor containing indices.
    /// \param depth     Specifies number of classes and the size of one-hot dimension.
    /// \param on_value  Specifies value that the locations in output tensor represented
    ///                  by indices in input take.
    /// \param off_value Specifies value that the locations in output tensor not
    /// represented
    ///                  by indices in input take.
    /// \param axis      Axis along which one-hot representation in added.
    OneHot(const Output<Node>& indices,
           const Output<Node>& depth,
           const Output<Node>& on_value,
           const Output<Node>& off_value,
           int64_t axis,
           NegativeIndicesMode mode = NegativeIndicesMode::IGNORE_NEGATIVE);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    /// \return The index of the one-hot axis.
    const int64_t& get_axis() const {
        return m_axis;
    }
    void set_axis(int64_t axis);

    /// \brief Sets the negative indices mode.
    void set_negative_indices_mode(NegativeIndicesMode mode) {
        m_negative_indices_mode = mode;
    }
    /// \return The negative indices mode.
    NegativeIndicesMode get_negative_indices_mode() const {
        return m_negative_indices_mode;
    }

protected:
    int64_t m_axis;
    NegativeIndicesMode m_negative_indices_mode;

private:
    friend void inline resolve_axis(OneHot* op);
};
}  // namespace v1
}  // namespace op

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v1::OneHot::NegativeIndicesMode& mode);

template <>
class OPENVINO_API AttributeAdapter<op::v1::OneHot::NegativeIndicesMode>
    : public EnumAttributeAdapterBase<op::v1::OneHot::NegativeIndicesMode> {
public:
    AttributeAdapter(op::v1::OneHot::NegativeIndicesMode& value)
        : EnumAttributeAdapterBase<op::v1::OneHot::NegativeIndicesMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v1::OneHot::NegativeIndicesMode>");
    ~AttributeAdapter() override;
};

}  // namespace ov
