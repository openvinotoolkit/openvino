// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>

#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace op {
namespace util {
struct VariableInfo {
    Shape data_shape;
    element::Type data_type;
    std::string variable_id;

    inline bool operator==(const VariableInfo& other) const {
        return data_shape == other.data_shape && data_type == other.data_type && variable_id == other.variable_id;
    }
};

class OPENVINO_API Variable {
public:
    using Ptr = std::shared_ptr<Variable>;
    Variable() = default;

    explicit Variable(VariableInfo variable_info) : m_info(std::move(variable_info)) {}

    VariableInfo get_info() const {
        return m_info;
    }
    void update(const VariableInfo& variable_info) {
        m_info = variable_info;
    }

private:
    VariableInfo m_info;
};
using VariableVector = std::vector<Variable::Ptr>;

}  // namespace util
}  // namespace op
template <>
class OPENVINO_API AttributeAdapter<std::shared_ptr<op::util::Variable>>
    : public DirectValueAccessor<std::shared_ptr<op::util::Variable>> {
public:
    explicit AttributeAdapter(std::shared_ptr<op::util::Variable>& value)
        : DirectValueAccessor<std::shared_ptr<op::util::Variable>>(value) {}

    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<std::shared_ptr<Variable>>", 0};
    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
};
}  // namespace ov
