// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/attr_types.hpp"

#include <cctype>
#include <map>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/check.hpp"
#include "ngraph/enum_names.hpp"

namespace ov {

template <>
NGRAPH_API EnumNames<ngraph::op::PadMode>& EnumNames<ngraph::op::PadMode>::get() {
    static auto enum_names = EnumNames<ngraph::op::PadMode>("ngraph::op::PadMode",
                                                            {{"constant", ngraph::op::PadMode::CONSTANT},
                                                             {"edge", ngraph::op::PadMode::EDGE},
                                                             {"reflect", ngraph::op::PadMode::REFLECT},
                                                             {"symmetric", ngraph::op::PadMode::SYMMETRIC}});
    return enum_names;
}

template <>
NGRAPH_API EnumNames<ngraph::op::PadType>& EnumNames<ngraph::op::PadType>::get() {
    static auto enum_names = EnumNames<ngraph::op::PadType>("ngraph::op::PadType",
                                                            {{"explicit", ngraph::op::PadType::EXPLICIT},
                                                             {"same_lower", ngraph::op::PadType::SAME_LOWER},
                                                             {"same_upper", ngraph::op::PadType::SAME_UPPER},
                                                             {"valid", ngraph::op::PadType::VALID}});
    return enum_names;
}

template <>
NGRAPH_API EnumNames<ngraph::op::RoundingType>& EnumNames<ngraph::op::RoundingType>::get() {
    static auto enum_names = EnumNames<ngraph::op::RoundingType>(
        "ngraph::op::RoundingType",
        {{"floor", ngraph::op::RoundingType::FLOOR}, {"ceil", ngraph::op::RoundingType::CEIL}});
    return enum_names;
}

template <>
NGRAPH_API EnumNames<ngraph::op::AutoBroadcastType>& EnumNames<ngraph::op::AutoBroadcastType>::get() {
    static auto enum_names =
        EnumNames<ngraph::op::AutoBroadcastType>("ngraph::op::AutoBroadcastType",
                                                 {{"none", ngraph::op::AutoBroadcastType::NONE},
                                                  {"explicit", ngraph::op::AutoBroadcastType::EXPLICIT},
                                                  {"numpy", ngraph::op::AutoBroadcastType::NUMPY},
                                                  {"pdpd", ngraph::op::AutoBroadcastType::PDPD}});
    return enum_names;
}

template <>
NGRAPH_API EnumNames<ngraph::op::BroadcastType>& EnumNames<ngraph::op::BroadcastType>::get() {
    static auto enum_names =
        EnumNames<ngraph::op::BroadcastType>("ngraph::op::BroadcastType",
                                             {{"explicit", ngraph::op::BroadcastType::EXPLICIT},
                                              {"none", ngraph::op::BroadcastType::NONE},
                                              {"numpy", ngraph::op::BroadcastType::NUMPY},
                                              {"pdpd", ngraph::op::BroadcastType::PDPD},
                                              {"bidirectional", ngraph::op::BroadcastType::BIDIRECTIONAL}});
    return enum_names;
}

template <>
NGRAPH_API EnumNames<ngraph::op::EpsMode>& EnumNames<ngraph::op::EpsMode>::get() {
    static auto enum_names =
        EnumNames<ngraph::op::EpsMode>("ngraph::op::EpsMode",
                                       {{"add", ngraph::op::EpsMode::ADD}, {"max", ngraph::op::EpsMode::MAX}});
    return enum_names;
}

template <>
NGRAPH_API EnumNames<ngraph::op::TopKSortType>& EnumNames<ngraph::op::TopKSortType>::get() {
    static auto enum_names = EnumNames<ngraph::op::TopKSortType>("ngraph::op::TopKSortType",
                                                                 {{"none", ngraph::op::TopKSortType::NONE},
                                                                  {"index", ngraph::op::TopKSortType::SORT_INDICES},
                                                                  {"value", ngraph::op::TopKSortType::SORT_VALUES}});
    return enum_names;
}
template <>
NGRAPH_API EnumNames<ngraph::op::TopKMode>& EnumNames<ngraph::op::TopKMode>::get() {
    static auto enum_names =
        EnumNames<ngraph::op::TopKMode>("ngraph::op::TopKMode",
                                        {{"min", ngraph::op::TopKMode::MIN}, {"max", ngraph::op::TopKMode::MAX}});
    return enum_names;
}

bool AttributeAdapter<ngraph::op::AutoBroadcastSpec>::visit_attributes(AttributeVisitor& visitor) {
    // Maintain back-compatibility
    std::string name = visitor.finish_structure();
    visitor.on_attribute(name, m_ref.m_type);
    visitor.start_structure(name);
    if (m_ref.m_type == ngraph::op::AutoBroadcastType::PDPD) {
        visitor.on_attribute("auto_broadcast_axis", m_ref.m_axis);
    }
    return true;
}

bool AttributeAdapter<ngraph::op::BroadcastModeSpec>::visit_attributes(AttributeVisitor& visitor) {
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

template <>
NGRAPH_API EnumNames<ngraph::op::RecurrentSequenceDirection>& EnumNames<ngraph::op::RecurrentSequenceDirection>::get() {
    static auto enum_names = EnumNames<ngraph::op::RecurrentSequenceDirection>(
        "ngraph::op::RecurrentSequenceDirection",
        {{"forward", ngraph::op::RecurrentSequenceDirection::FORWARD},
         {"reverse", ngraph::op::RecurrentSequenceDirection::REVERSE},
         {"bidirectional", ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL}});
    return enum_names;
}

std::ostream& op::operator<<(std::ostream& s, const ngraph::op::PadMode& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ngraph::op::PadType& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ngraph::op::RoundingType& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ngraph::op::BroadcastType& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ngraph::op::AutoBroadcastType& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ngraph::op::EpsMode& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ngraph::op::TopKSortType& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ngraph::op::TopKMode& type) {
    return s << as_string(type);
}

op::AutoBroadcastType op::AutoBroadcastSpec::type_from_string(const std::string& type) const {
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

std::ostream& op::operator<<(std::ostream& s, const ngraph::op::RecurrentSequenceDirection& direction) {
    return s << as_string(direction);
}
}  // namespace ov
