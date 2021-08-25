// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph {
namespace reduction {
enum class Type {
    SUM,
    PROD,
    MIN,
    MAX,
};

NGRAPH_API
std::ostream& operator<<(std::ostream& out, const Type& obj);
}  // namespace reduction

}  // namespace ngraph

namespace ov {

template <>
class NGRAPH_API AttributeAdapter<ngraph::reduction::Type> : public EnumAttributeAdapterBase<ngraph::reduction::Type> {
public:
    AttributeAdapter(ngraph::reduction::Type& value) : EnumAttributeAdapterBase<ngraph::reduction::Type>(value) {}

    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<reduction::Type>", 0};
    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
};

}  // namespace ov
