//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <string>
#include <utility>

#include "ngraph/check.hpp"

namespace ngraph
{
    /// Uses a pairings defined by EnumTypes::get() to convert between strings
    /// and enum values.
    template <typename EnumType>
    class EnumNames
    {
    public:
        /// Converts strings to enum values
        static EnumType as_enum(const std::string& name)
        {
            for (auto p : get().m_string_enums)
            {
                if (p.first == name)
                {
                    return p.second;
                }
            }
            NGRAPH_CHECK(false, "\"", name, "\"", " is not a member of enum ", get().m_enum_name);
        }

        /// Converts enum values to strings
        static const std::string& as_string(EnumType e)
        {
            for (auto& p : get().m_string_enums)
            {
                if (p.second == e)
                {
                    return p.first;
                }
            }
            NGRAPH_CHECK(false, " invalid member of enum ", get().m_enum_name);
        }

    private:
        /// Creates the mapping.
        EnumNames(const std::string& enum_name,
                  const std::vector<std::pair<std::string, EnumType>> string_enums)
            : m_enum_name(enum_name)
            , m_string_enums(string_enums)
        {
        }

        /// Must be defined to returns a singleton for each supported enum class
        static EnumNames<EnumType>& get();

        const std::string m_enum_name;
        std::vector<std::pair<std::string, EnumType>> m_string_enums;
    };

    /// Returns the enum value matching the string
    template <typename Type, typename Value>
    typename std::enable_if<std::is_convertible<Value, std::string>::value, Type>::type
        as_enum(const Value& value)
    {
        return EnumNames<Type>::as_enum(value);
    }

    /// Returns the string matching the enum value
    template <typename Value>
    const std::string& as_string(Value value)
    {
        return EnumNames<Value>::as_string(value);
    }
}
