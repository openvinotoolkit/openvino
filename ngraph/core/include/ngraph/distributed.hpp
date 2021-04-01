//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace reduction
    {
        enum class Type
        {
            SUM,
            PROD,
            MIN,
            MAX,
        };

        NGRAPH_API
        std::ostream& operator<<(std::ostream& out, const Type& obj);
    }

    template <>
    class NGRAPH_API AttributeAdapter<reduction::Type>
        : public EnumAttributeAdapterBase<reduction::Type>
    {
    public:
        AttributeAdapter(reduction::Type& value)
            : EnumAttributeAdapterBase<reduction::Type>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<reduction::Type>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
}
