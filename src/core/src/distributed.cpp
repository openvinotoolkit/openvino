// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/distributed.hpp"

#include "ngraph/log.hpp"
#include "ngraph/type.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START
using namespace ngraph;

namespace ov {
template <>
NGRAPH_API EnumNames<ngraph::reduction::Type>& EnumNames<ngraph::reduction::Type>::get() {
    static auto enum_names = ov::EnumNames<ngraph::reduction::Type>("reduction::Type",
                                                                    {{"SUM", ngraph::reduction::Type::SUM},
                                                                     {"PROD", ngraph::reduction::Type::PROD},
                                                                     {"MIN", ngraph::reduction::Type::MIN},
                                                                     {"MAX", ngraph::reduction::Type::MAX}});
    return enum_names;
}
BWDCMP_RTTI_DEFINITION(AttributeAdapter<ngraph::reduction::Type>);
}  // namespace ov

std::ostream& ngraph::reduction::operator<<(std::ostream& out, const ngraph::reduction::Type& obj) {
    return out << as_string(obj);
}
