// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {

namespace v13 {
/// \brief NMSRotated operation
///
class OPENVINO_API NMSRotated : public Op {
public:
    OPENVINO_OP("NMSRotated", "opset13", op::Op);
    enum class BoxEncodingType { CORNER, CENTER };

    NMSRotated() = default;

    /// \brief Constructs a NMSRotated operation with default value in the last.
    ///        input.
    ///
    /// \param boxes Node containing the box coordinates
    /// \param scores Node containing the box scores
    /// \param max_output_boxes_per_class Node containing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node containing intersection over union threshold
    /// \param score_threshold Node containing minimum score threshold
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    /// \param clockwise Specifies the direction of the rotation
    NMSRotated(const Output<Node>& boxes,
               const Output<Node>& scores,
               const Output<Node>& max_output_boxes_per_class,
               const Output<Node>& iou_threshold,
               const Output<Node>& score_threshold,
               const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
               const bool sort_result_descending = true,
               const ov::element::Type& output_type = ov::element::i64,
               const bool clockwise = true);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    BoxEncodingType get_box_encoding() const {
        return m_box_encoding;
    }
    void set_box_encoding(const BoxEncodingType box_encoding) {
        m_box_encoding = box_encoding;
    }

    bool get_sort_result_descending() const {
        return m_sort_result_descending;
    }
    void set_sort_result_descending(const bool sort_result_descending) {
        m_sort_result_descending = sort_result_descending;
    }

    element::Type get_output_type() const {
        return m_output_type;
    }
    void set_output_type(const element::Type& output_type) {
        m_output_type = output_type;
    }

    bool get_clockwise() const {
        return m_clockwise;
    }
    void set_clockwise(const bool clockwise) {
        m_clockwise = clockwise;
    }

    using Node::set_output_type;

    // Temporary evaluate, for testing purpose
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

protected:
    BoxEncodingType m_box_encoding = BoxEncodingType::CORNER;
    bool m_sort_result_descending = true;
    ov::element::Type m_output_type = ov::element::i64;
    bool m_clockwise = true;
};
}  // namespace v13
}  // namespace op

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v13::NMSRotated::BoxEncodingType& type);

template <>
class OPENVINO_API AttributeAdapter<op::v13::NMSRotated::BoxEncodingType>
    : public EnumAttributeAdapterBase<op::v13::NMSRotated::BoxEncodingType> {
public:
    AttributeAdapter(op::v13::NMSRotated::BoxEncodingType& value)
        : EnumAttributeAdapterBase<op::v13::NMSRotated::BoxEncodingType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::v13::NMSRotated::BoxEncodingType>");
};

}  // namespace ov
