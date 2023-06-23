// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <cstddef>
#include <memory>
#include <string>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/deprecated.hpp"
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

NGRAPH_API_DEPRECATED NGRAPH_API std::ostream& operator<<(std::ostream& out, const Type& obj);
}  // namespace reduction

}  // namespace ngraph

namespace ov {

template <>
class NGRAPH_API_DEPRECATED NGRAPH_API AttributeAdapter<ngraph::reduction::Type>
    : public EnumAttributeAdapterBase<ngraph::reduction::Type> {
public:
    AttributeAdapter(ngraph::reduction::Type& value) : EnumAttributeAdapterBase<ngraph::reduction::Type>(value) {}

    OPENVINO_RTTI("AttributeAdapter<reduction::Type>");
};

}  // namespace ov
