// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
namespace util {
/// \brief Base class for operations NmsBase and MatrixNms
///
class NGRAPH_API NmsBase : public Op {
public:
    NGRAPH_RTTI_DECLARATION;
    enum class SortResultType {
        CLASSID,  // sort selected boxes by class id (ascending) in each batch element
        SCORE,    // sort selected boxes by score (descending) in each batch element
        NONE      // do not guarantee the order in each batch element
    };

    NmsBase() = delete;

    /// \brief Constructs a NmsBase operation
    ///
    /// \param output_type Specifies the output tensor type
    /// \param nms_top_k Specifies maximum number of boxes to be selected per
    /// class, -1 meaning to keep all boxes
    /// \param keep_top_k Specifies maximum number of boxes to be selected per
    /// batch element, -1 meaning to keep all boxes
    NmsBase(ngraph::element::Type& output_type, int& nms_top_k, int& keep_top_k);

    /// \brief Constructs a NmsBase operation
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param output_type Specifies the output tensor type
    /// \param nms_top_k Specifies maximum number of boxes to be selected per
    /// class, -1 meaning to keep all boxes
    /// \param keep_top_k Specifies maximum number of boxes to be selected per
    /// batch element, -1 meaning to keep all boxes
    NmsBase(const Output<Node>& boxes,
            const Output<Node>& scores,
            ngraph::element::Type& output_type,
            int& nms_top_k,
            int& keep_top_k);

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
    ngraph::element::Type& m_output_type;
    int& m_nms_top_k;
    int& m_keep_top_k;
    virtual void validate();
};
}  // namespace util
}  // namespace op

NGRAPH_API
std::ostream& operator<<(std::ostream& s, const op::util::NmsBase::SortResultType& type);
}  // namespace ngraph

namespace ov {

template <>
class NGRAPH_API AttributeAdapter<ngraph::op::util::NmsBase::SortResultType>
    : public EnumAttributeAdapterBase<ngraph::op::util::NmsBase::SortResultType> {
public:
    AttributeAdapter(ngraph::op::util::NmsBase::SortResultType& value)
        : EnumAttributeAdapterBase<ngraph::op::util::NmsBase::SortResultType>(value) {}

    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::util::NmsBase::SortResultType>", 1};
    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
};

}  // namespace ov
