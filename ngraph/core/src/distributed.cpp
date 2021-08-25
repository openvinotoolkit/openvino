// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/distributed.hpp"

#include "ngraph/log.hpp"
#include "ngraph/type.hpp"

namespace ov {
template <>
EnumNames<ngraph::reduction::Type>& EnumNames<ngraph::reduction::Type>::get() {
    static auto enum_names = ov::EnumNames<ngraph::reduction::Type>("reduction::Type",
                                                                    {{"SUM", ngraph::reduction::Type::SUM},
                                                                     {"PROD", ngraph::reduction::Type::PROD},
                                                                     {"MIN", ngraph::reduction::Type::MIN},
                                                                     {"MAX", ngraph::reduction::Type::MAX}});
    return enum_names;
}
constexpr DiscreteTypeInfo AttributeAdapter<ngraph::reduction::Type>::type_info;
}  // namespace ov

std::ostream& ngraph::reduction::operator<<(std::ostream& out, const ngraph::reduction::Type& obj) {
    return out << as_string(obj);
}
