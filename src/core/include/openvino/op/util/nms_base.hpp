// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Base class for operations NmsBase and MatrixNms
///
class OPENVINO_API NmsBase : public Op {
public:
    OPENVINO_OP("NmsBase", "util");
    enum class SortResultType {
        CLASSID,  // sort selected boxes by class id (ascending) in each batch element
        SCORE,    // sort selected boxes by score (descending) in each batch element
        NONE      // do not guarantee the order in each batch element
    };

    /// \brief Constructs a NmsBase operation
    // TODO: remove the reference to inherit, and remove the construtors here.
    NmsBase() = delete;
    NmsBase(ngraph::element::Type& output_type, int& nms_top_k, int& keep_top_k);

    /// \brief Constructs a NmsBase operation
    ///
    /// \param arguments Node producing the box coordinates and scores, etc.
    /// \param output_type Specifies the output tensor type
    /// \param nms_top_k Specifies maximum number of boxes to be selected per
    /// class, -1 meaning to keep all boxes
    /// \param keep_top_k Specifies maximum number of boxes to be selected per
    /// batch element, -1 meaning to keep all boxes
    NmsBase(const OutputVector& arguments, element::Type& output_type, int& nms_top_k, int& keep_top_k);

    void validate_and_infer_types() override;

    const element::Type& get_output_type() const {
        return m_output_type;
    }
    void set_output_type(const element::Type& output_type) {
        m_output_type = output_type;
    }
    using Node::set_output_type;

    int get_nms_top_k() const {
        return m_nms_top_k;
    }

    int get_keep_top_k() const {
        return m_keep_top_k;
    }

protected:
    element::Type& m_output_type;
    int& m_nms_top_k;
    int& m_keep_top_k;
    virtual bool validate();
};
}  // namespace util
}  // namespace op

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::util::NmsBase::SortResultType& type);

template <>
class OPENVINO_API AttributeAdapter<op::util::NmsBase::SortResultType>
    : public EnumAttributeAdapterBase<op::util::NmsBase::SortResultType> {
public:
    AttributeAdapter(op::util::NmsBase::SortResultType& value)
        : EnumAttributeAdapterBase<op::util::NmsBase::SortResultType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::util::NmsBase::SortResultType>");
    BWDCMP_RTTI_DECLARATION;
};

}  // namespace ov
