// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/distributed.hpp"
#include "ngraph/log.hpp"
#include "ngraph/type.hpp"

using namespace ngraph;

namespace ngraph
{
    template <>
    EnumNames<reduction::Type>& EnumNames<reduction::Type>::get()
    {
        static auto enum_names = EnumNames<reduction::Type>("reduction::Type",
                                                            {{"SUM", reduction::Type::SUM},
                                                             {"PROD", reduction::Type::PROD},
                                                             {"MIN", reduction::Type::MIN},
                                                             {"MAX", reduction::Type::MAX}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<reduction::Type>::type_info;
} // namespace ngraph

std::ostream& reduction::operator<<(std::ostream& out, const reduction::Type& obj)
{
    return out << as_string(obj);
}
