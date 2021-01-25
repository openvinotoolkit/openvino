// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(Eltwise) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"Eltwise", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    enum class EltwiseType {Sum, Prod, Max, Sub, Min, Div};

    Eltwise(const Output<Node>& data1,
            const Output<Node>& data2,
            const EltwiseType eltwise_type,
            const element::Type output_type = element::undefined);

    bool visit_attributes(AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    EltwiseType eltwise_type;

private:
    EltwiseType type_from_string(const std::string &eltwise_type) const { return as_enum<EltwiseType>(eltwise_type); }
    element::Type m_output_type;
};

} // namespace op
std::ostream &operator<<(std::ostream &s, const op::Eltwise::EltwiseType &type);

template <>
class NGRAPH_API AttributeAdapter<op::Eltwise::EltwiseType>
    : public EnumAttributeAdapterBase<op::Eltwise::EltwiseType> {
public:
  AttributeAdapter(op::Eltwise::EltwiseType &value)
      : EnumAttributeAdapterBase<op::Eltwise::EltwiseType>(value) {}

  static constexpr DiscreteTypeInfo type_info{
      "AttributeAdapter<op::Eltwise::EltwiseType>", 1};
  const DiscreteTypeInfo &get_type_info() const override { return type_info; }
};
} // namespace ngraph
