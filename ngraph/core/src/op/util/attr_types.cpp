// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/attr_types.hpp"

#include <cctype>
#include <map>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/check.hpp"
#include "ngraph/enum_names.hpp"

const ngraph::op::AutoBroadcastSpec ngraph::op::AutoBroadcastSpec::NUMPY(AutoBroadcastType::NUMPY, 0);
const ngraph::op::AutoBroadcastSpec ngraph::op::AutoBroadcastSpec::NONE{AutoBroadcastType::NONE, 0};

template <>
NGRAPH_API ov::EnumNames<ngraph::op::PadMode>& ov::EnumNames<ngraph::op::PadMode>::get() {
    static auto enum_names = ov::EnumNames<ngraph::op::PadMode>("ngraph::op::PadMode",
                                                                {{"constant", ngraph::op::PadMode::CONSTANT},
                                                                 {"edge", ngraph::op::PadMode::EDGE},
                                                                 {"reflect", ngraph::op::PadMode::REFLECT},
                                                                 {"symmetric", ngraph::op::PadMode::SYMMETRIC}});
    return enum_names;
}

constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ngraph::op::PadMode>::type_info;

std::ostream& ngraph::op::operator<<(std::ostream& s, const ngraph::op::PadMode& type) {
    return s << as_string(type);
}
template <>
NGRAPH_API ov::EnumNames<ngraph::op::PadType>& ov::EnumNames<ngraph::op::PadType>::get() {
    static auto enum_names = ov::EnumNames<ngraph::op::PadType>("ngraph::op::PadType",
                                                                {{"explicit", ngraph::op::PadType::EXPLICIT},
                                                                 {"same_lower", ngraph::op::PadType::SAME_LOWER},
                                                                 {"same_upper", ngraph::op::PadType::SAME_UPPER},
                                                                 {"valid", ngraph::op::PadType::VALID}});
    return enum_names;
}

constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ngraph::op::PadType>::type_info;

std::ostream& ngraph::op::operator<<(std::ostream& s, const ngraph::op::PadType& type) {
    return s << as_string(type);
}
template <>
NGRAPH_API ov::EnumNames<ngraph::op::RoundingType>& ov::EnumNames<ngraph::op::RoundingType>::get() {
    static auto enum_names = ov::EnumNames<ngraph::op::RoundingType>(
        "ngraph::op::RoundingType",
        {{"floor", ngraph::op::RoundingType::FLOOR}, {"ceil", ngraph::op::RoundingType::CEIL}});
    return enum_names;
}

constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ngraph::op::RoundingType>::type_info;

std::ostream& ngraph::op::operator<<(std::ostream& s, const ngraph::op::RoundingType& type) {
    return s << as_string(type);
}

template <>
NGRAPH_API ov::EnumNames<ngraph::op::AutoBroadcastType>& ov::EnumNames<ngraph::op::AutoBroadcastType>::get() {
    static auto enum_names =
        ov::EnumNames<ngraph::op::AutoBroadcastType>("ngraph::op::AutoBroadcastType",
                                                     {{"none", ngraph::op::AutoBroadcastType::NONE},
                                                      {"explicit", ngraph::op::AutoBroadcastType::EXPLICIT},
                                                      {"numpy", ngraph::op::AutoBroadcastType::NUMPY},
                                                      {"pdpd", ngraph::op::AutoBroadcastType::PDPD}});
    return enum_names;
}
constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ngraph::op::AutoBroadcastType>::type_info;

template <>
NGRAPH_API ov::EnumNames<ngraph::op::BroadcastType>& ov::EnumNames<ngraph::op::BroadcastType>::get() {
    static auto enum_names =
        ov::EnumNames<ngraph::op::BroadcastType>("ngraph::op::BroadcastType",
                                                 {{"none", ngraph::op::BroadcastType::NONE},
                                                  {"numpy", ngraph::op::BroadcastType::NUMPY},
                                                  {"explicit", ngraph::op::BroadcastType::EXPLICIT},
                                                  {"pdpd", ngraph::op::BroadcastType::PDPD},
                                                  {"bidirectional", ngraph::op::BroadcastType::BIDIRECTIONAL}});
    return enum_names;
}

std::ostream& ngraph::op::operator<<(std::ostream& s, const ngraph::op::BroadcastType& type) {
    return s << as_string(type);
}

constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ngraph::op::BroadcastType>::type_info;

std::ostream& ngraph::op::operator<<(std::ostream& s, const ngraph::op::AutoBroadcastType& type) {
    return s << as_string(type);
}
template <>
NGRAPH_API ov::EnumNames<ngraph::op::EpsMode>& ov::EnumNames<ngraph::op::EpsMode>::get() {
    static auto enum_names =
        ov::EnumNames<ngraph::op::EpsMode>("ngraph::op::EpsMode",
                                           {{"add", ngraph::op::EpsMode::ADD}, {"max", ngraph::op::EpsMode::MAX}});
    return enum_names;
}

constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ngraph::op::EpsMode>::type_info;

std::ostream& ngraph::op::operator<<(std::ostream& s, const ngraph::op::EpsMode& type) {
    return s << as_string(type);
}

template <>
NGRAPH_API ov::EnumNames<ngraph::op::TopKSortType>& ov::EnumNames<ngraph::op::TopKSortType>::get() {
    static auto enum_names =
        ov::EnumNames<ngraph::op::TopKSortType>("ngraph::op::TopKSortType",
                                                {{"none", ngraph::op::TopKSortType::NONE},
                                                 {"index", ngraph::op::TopKSortType::SORT_INDICES},
                                                 {"value", ngraph::op::TopKSortType::SORT_VALUES}});
    return enum_names;
}
template <>
NGRAPH_API ov::EnumNames<ngraph::op::TopKMode>& ov::EnumNames<ngraph::op::TopKMode>::get() {
    static auto enum_names =
        ov::EnumNames<ngraph::op::TopKMode>("ngraph::op::TopKMode",
                                            {{"min", ngraph::op::TopKMode::MIN}, {"max", ngraph::op::TopKMode::MAX}});
    return enum_names;
}

constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ngraph::op::TopKSortType>::type_info;
constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ngraph::op::TopKMode>::type_info;

std::ostream& ngraph::op::operator<<(std::ostream& s, const ngraph::op::TopKSortType& type) {
    return s << as_string(type);
}

std::ostream& ngraph::op::operator<<(std::ostream& s, const ngraph::op::TopKMode& type) {
    return s << as_string(type);
}

ngraph::op::AutoBroadcastType ngraph::op::AutoBroadcastSpec::type_from_string(const std::string& type) const {
    auto lowercase_type = type;
    std::transform(lowercase_type.begin(), lowercase_type.end(), lowercase_type.begin(), [](char c) {
        return std::tolower(c);
    });

    static const std::map<std::string, AutoBroadcastType> allowed_values = {{"none", AutoBroadcastType::NONE},
                                                                            {"numpy", AutoBroadcastType::NUMPY},
                                                                            {"pdpd", AutoBroadcastType::PDPD},
                                                                            {"explicit", AutoBroadcastType::EXPLICIT}};

    NGRAPH_CHECK(allowed_values.count(lowercase_type) > 0, "Invalid 'type' value passed in.");

    return allowed_values.at(lowercase_type);
}

bool ov::AttributeAdapter<ngraph::op::AutoBroadcastSpec>::visit_attributes(AttributeVisitor& visitor) {
    // Maintain back-compatibility
    std::string name = visitor.finish_structure();
    visitor.on_attribute(name, m_ref.m_type);
    visitor.start_structure(name);
    if (m_ref.m_type == ngraph::op::AutoBroadcastType::PDPD) {
        visitor.on_attribute("auto_broadcast_axis", m_ref.m_axis);
    }
    return true;
}

constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ngraph::op::AutoBroadcastSpec>::type_info;

bool ov::AttributeAdapter<ngraph::op::BroadcastModeSpec>::visit_attributes(AttributeVisitor& visitor) {
    // Maintain back-compatibility
    std::string name = visitor.finish_structure();
    visitor.on_attribute(name, m_ref.m_type);
    visitor.start_structure(name);
    if (m_ref.m_type == ngraph::op::BroadcastType::PDPD) {
        visitor.start_structure(name);
        visitor.on_attribute("axis", m_ref.m_axis);
        visitor.finish_structure();
    }
    return true;
}

constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ngraph::op::BroadcastModeSpec>::type_info;

NGRAPH_API
constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ngraph::op::RecurrentSequenceDirection>::type_info;

std::ostream& ngraph::op::operator<<(std::ostream& s, const ngraph::op::RecurrentSequenceDirection& direction) {
    return s << as_string(direction);
}
template <>
NGRAPH_API ov::EnumNames<ngraph::op::RecurrentSequenceDirection>&
ov::EnumNames<ngraph::op::RecurrentSequenceDirection>::get() {
    static auto enum_names = ov::EnumNames<ngraph::op::RecurrentSequenceDirection>(
        "ngraph::op::RecurrentSequenceDirection",
        {{"forward", ngraph::op::RecurrentSequenceDirection::FORWARD},
         {"reverse", ngraph::op::RecurrentSequenceDirection::REVERSE},
         {"bidirectional", ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL}});
    return enum_names;
}
