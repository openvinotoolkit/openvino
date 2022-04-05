// Copyright (C) 2018-2022 Intel Corporation
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

NGRAPH_SUPPRESS_DEPRECATED_START
NGRAPH_API
std::ostream& operator<<(std::ostream& out, const Type& obj);
NGRAPH_SUPPRESS_DEPRECATED_END
}  // namespace reduction

}  // namespace ngraph

namespace ov {

template <>
class NGRAPH_API AttributeAdapter<ngraph::reduction::Type> : public EnumAttributeAdapterBase<ngraph::reduction::Type> {
public:
    AttributeAdapter(ngraph::reduction::Type& value) : EnumAttributeAdapterBase<ngraph::reduction::Type>(value) {}

    OPENVINO_RTTI("AttributeAdapter<reduction::Type>");
    BWDCMP_RTTI_DECLARATION;
};

}  // namespace ov
