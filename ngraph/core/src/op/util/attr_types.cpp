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
#include <map>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/check.hpp"
#include "ngraph/enum_names.hpp"
#include "ngraph/op/util/attr_types.hpp"

using namespace ngraph;

const op::AutoBroadcastSpec op::AutoBroadcastSpec::NUMPY(AutoBroadcastType::NUMPY, 0);
const op::AutoBroadcastSpec op::AutoBroadcastSpec::NONE{AutoBroadcastType::NONE, 0};

namespace ngraph
{
    template <>
    NGRAPH_API EnumNames<op::PadMode>& EnumNames<op::PadMode>::get()
    {
        static auto enum_names = EnumNames<op::PadMode>("op::PadMode",
                                                        {{"CONSTANT", op::PadMode::CONSTANT},
                                                         {"EDGE", op::PadMode::EDGE},
                                                         {"REFLECT", op::PadMode::REFLECT},
                                                         {"SYMMETRIC", op::PadMode::SYMMETRIC}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::PadMode>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::PadMode& type)
    {
        return s << as_string(type);
    }
    template <>
    NGRAPH_API EnumNames<op::PadType>& EnumNames<op::PadType>::get()
    {
        static auto enum_names = EnumNames<op::PadType>("op::PadType",
                                                        {{"explicit", op::PadType::EXPLICIT},
                                                         {"same_lower", op::PadType::SAME_LOWER},
                                                         {"same_upper", op::PadType::SAME_UPPER},
                                                         {"valid", op::PadType::VALID}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::PadType>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::PadType& type)
    {
        return s << as_string(type);
    }
    template <>
    NGRAPH_API EnumNames<op::RoundingType>& EnumNames<op::RoundingType>::get()
    {
        static auto enum_names = EnumNames<op::RoundingType>(
            "op::RoundingType",
            {{"floor", op::RoundingType::FLOOR}, {"ceil", op::RoundingType::CEIL}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::RoundingType>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::RoundingType& type)
    {
        return s << as_string(type);
    }

    template <>
    NGRAPH_API EnumNames<op::AutoBroadcastType>& EnumNames<op::AutoBroadcastType>::get()
    {
        static auto enum_names =
            EnumNames<op::AutoBroadcastType>("op::AutoBroadcastType",
                                             {{"none", op::AutoBroadcastType::NONE},
                                              {"explicit", op::AutoBroadcastType::EXPLICIT},
                                              {"numpy", op::AutoBroadcastType::NUMPY},
                                              {"pdpd", op::AutoBroadcastType::PDPD}});
        return enum_names;
    }
    constexpr DiscreteTypeInfo AttributeAdapter<op::AutoBroadcastType>::type_info;

    template <>
    NGRAPH_API EnumNames<op::BroadcastType>& EnumNames<op::BroadcastType>::get()
    {
        static auto enum_names =
            EnumNames<op::BroadcastType>("op::BroadcastType",
                                         {{"NONE", op::BroadcastType::NONE},
                                          {"NUMPY", op::BroadcastType::NUMPY},
                                          {"EXPLICIT", op::BroadcastType::EXPLICIT},
                                          {"PDPD", op::BroadcastType::PDPD},
                                          {"BIDIRECTIONAL", op::BroadcastType::BIDIRECTIONAL}});
        return enum_names;
    }

    std::ostream& op::operator<<(std::ostream& s, const op::BroadcastType& type)
    {
        return s << as_string(type);
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::BroadcastType>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::AutoBroadcastType& type)
    {
        return s << as_string(type);
    }
    template <>
    NGRAPH_API EnumNames<op::EpsMode>& EnumNames<op::EpsMode>::get()
    {
        static auto enum_names = EnumNames<op::EpsMode>(
            "op::EpsMode", {{"add", op::EpsMode::ADD}, {"max", op::EpsMode::MAX}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::EpsMode>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::EpsMode& type)
    {
        return s << as_string(type);
    }
    template <>
    NGRAPH_API EnumNames<op::TopKSortType>& EnumNames<op::TopKSortType>::get()
    {
        static auto enum_names =
            EnumNames<op::TopKSortType>("op::TopKSortType",
                                        {{"none", op::TopKSortType::NONE},
                                         {"index", op::TopKSortType::SORT_INDICES},
                                         {"value", op::TopKSortType::SORT_VALUES}});
        return enum_names;
    }
    template <>
    NGRAPH_API EnumNames<op::TopKMode>& EnumNames<op::TopKMode>::get()
    {
        static auto enum_names = EnumNames<op::TopKMode>(
            "op::TopKMode", {{"min", op::TopKMode::MIN}, {"max", op::TopKMode::MAX}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::TopKSortType>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<op::TopKMode>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::TopKSortType& type)
    {
        return s << as_string(type);
    }

    std::ostream& op::operator<<(std::ostream& s, const op::TopKMode& type)
    {
        return s << as_string(type);
    }

    op::AutoBroadcastType op::AutoBroadcastSpec::type_from_string(const std::string& type) const
    {
        static const std::map<std::string, AutoBroadcastType> allowed_values = {
            {"NONE", AutoBroadcastType::NONE},
            {"NUMPY", AutoBroadcastType::NUMPY},
            {"PDPD", AutoBroadcastType::PDPD},
            {"EXPLICIT", AutoBroadcastType::EXPLICIT}};

        NGRAPH_CHECK(allowed_values.count(type) > 0, "Invalid 'type' value passed in.");

        return allowed_values.at(type);
    }

    bool AttributeAdapter<op::AutoBroadcastSpec>::visit_attributes(AttributeVisitor& visitor)
    {
        // Maintain back-compatibility
        std::string name = visitor.finish_structure();
        visitor.on_attribute(name, m_ref.m_type);
        visitor.start_structure(name);
        if (m_ref.m_type == op::AutoBroadcastType::PDPD)
        {
            visitor.on_attribute("auto_broadcast_axis", m_ref.m_axis);
        }
        return true;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::AutoBroadcastSpec>::type_info;

    bool AttributeAdapter<op::BroadcastModeSpec>::visit_attributes(AttributeVisitor& visitor)
    {
        // Maintain back-compatibility
        std::string name = visitor.finish_structure();
        visitor.on_attribute(name, m_ref.m_type);
        visitor.start_structure(name);
        if (m_ref.m_type == op::BroadcastType::PDPD)
        {
            visitor.start_structure(name);
            visitor.on_attribute("axis", m_ref.m_axis);
            visitor.finish_structure();
        }
        return true;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::BroadcastModeSpec>::type_info;

    NGRAPH_API
    constexpr DiscreteTypeInfo AttributeAdapter<op::RecurrentSequenceDirection>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::RecurrentSequenceDirection& direction)
    {
        return s << as_string(direction);
    }
    template <>
    NGRAPH_API EnumNames<op::RecurrentSequenceDirection>&
        EnumNames<op::RecurrentSequenceDirection>::get()
    {
        static auto enum_names = EnumNames<op::RecurrentSequenceDirection>(
            "op::RecurrentSequenceDirection",
            {{"forward", op::RecurrentSequenceDirection::FORWARD},
             {"reverse", op::RecurrentSequenceDirection::REVERSE},
             {"bidirectional", op::RecurrentSequenceDirection::BIDIRECTIONAL}});
        return enum_names;
    }
}
