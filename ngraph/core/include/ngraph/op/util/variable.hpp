// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>

#include "ngraph/partial_shape.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph {
struct VariableInfo {
    PartialShape data_shape;
    element::Type data_type;
    std::string variable_id;

    inline bool operator==(const VariableInfo& other) const {
        return data_shape == other.data_shape && data_type == other.data_type && variable_id == other.variable_id;
    }
};

class NGRAPH_API Variable {
public:
    Variable() = default;

    explicit Variable(const VariableInfo& variable_info) : m_info(variable_info) {}

    VariableInfo get_info() const {
        return m_info;
    }
    void update(const VariableInfo& variable_info) {
        m_info = variable_info;
    }

private:
    VariableInfo m_info;
};
using VariablePtr = std::shared_ptr<Variable>;
using VariableVector = std::vector<VariablePtr>;
}  // namespace ngraph

namespace ov {

template <>
class NGRAPH_API AttributeAdapter<std::shared_ptr<ngraph::Variable>>
    : public DirectValueAccessor<std::shared_ptr<ngraph::Variable>> {
public:
    explicit AttributeAdapter(std::shared_ptr<ngraph::Variable>& value)
        : DirectValueAccessor<std::shared_ptr<ngraph::Variable>>(value) {}

    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<std::shared_ptr<Variable>>", 0};
    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
};

}  // namespace ov
