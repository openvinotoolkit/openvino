// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/attr_types.hpp"

#include <cctype>
#include <map>

#include "openvino/core/except.hpp"

namespace ov {

template <>
OPENVINO_API EnumNames<ov::op::PadMode>& EnumNames<ov::op::PadMode>::get() {
    static auto enum_names = EnumNames<ov::op::PadMode>("ov::op::PadMode",
                                                        {{"constant", ov::op::PadMode::CONSTANT},
                                                         {"edge", ov::op::PadMode::EDGE},
                                                         {"reflect", ov::op::PadMode::REFLECT},
                                                         {"symmetric", ov::op::PadMode::SYMMETRIC}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<ov::op::FillMode>& EnumNames<ov::op::FillMode>::get() {
    static auto enum_names =
        EnumNames<ov::op::FillMode>("ov::op::FillMode",
                                    {{"zero", ov::op::FillMode::ZERO}, {"lowest", ov::op::FillMode::LOWEST}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<ov::op::PadType>& EnumNames<ov::op::PadType>::get() {
    static auto enum_names = EnumNames<ov::op::PadType>("ov::op::PadType",
                                                        {{"explicit", ov::op::PadType::EXPLICIT},
                                                         {"same_lower", ov::op::PadType::SAME_LOWER},
                                                         {"same_upper", ov::op::PadType::SAME_UPPER},
                                                         {"valid", ov::op::PadType::VALID}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<ov::op::RoundingType>& EnumNames<ov::op::RoundingType>::get() {
    static auto enum_names = EnumNames<ov::op::RoundingType>("ov::op::RoundingType",
                                                             {{"floor", ov::op::RoundingType::FLOOR},
                                                              {"ceil", ov::op::RoundingType::CEIL},
                                                              {"ceil_torch", ov::op::RoundingType::CEIL_TORCH}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<ov::op::AutoBroadcastType>& EnumNames<ov::op::AutoBroadcastType>::get() {
    static auto enum_names = EnumNames<ov::op::AutoBroadcastType>("ov::op::AutoBroadcastType",
                                                                  {{"none", ov::op::AutoBroadcastType::NONE},
                                                                   {"explicit", ov::op::AutoBroadcastType::EXPLICIT},
                                                                   {"numpy", ov::op::AutoBroadcastType::NUMPY},
                                                                   {"pdpd", ov::op::AutoBroadcastType::PDPD}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<ov::op::BroadcastType>& EnumNames<ov::op::BroadcastType>::get() {
    static auto enum_names =
        EnumNames<ov::op::BroadcastType>("ov::op::BroadcastType",
                                         {{"explicit", ov::op::BroadcastType::EXPLICIT},
                                          {"none", ov::op::BroadcastType::NONE},
                                          {"numpy", ov::op::BroadcastType::NUMPY},
                                          {"pdpd", ov::op::BroadcastType::PDPD},
                                          {"bidirectional", ov::op::BroadcastType::BIDIRECTIONAL}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<ov::op::EpsMode>& EnumNames<ov::op::EpsMode>::get() {
    static auto enum_names =
        EnumNames<ov::op::EpsMode>("ov::op::EpsMode", {{"add", ov::op::EpsMode::ADD}, {"max", ov::op::EpsMode::MAX}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<ov::op::TopKSortType>& EnumNames<ov::op::TopKSortType>::get() {
    static auto enum_names = EnumNames<ov::op::TopKSortType>("ov::op::TopKSortType",
                                                             {{"none", ov::op::TopKSortType::NONE},
                                                              {"index", ov::op::TopKSortType::SORT_INDICES},
                                                              {"value", ov::op::TopKSortType::SORT_VALUES}});
    return enum_names;
}
template <>
OPENVINO_API EnumNames<ov::op::TopKMode>& EnumNames<ov::op::TopKMode>::get() {
    static auto enum_names =
        EnumNames<ov::op::TopKMode>("ov::op::TopKMode",
                                    {{"min", ov::op::TopKMode::MIN}, {"max", ov::op::TopKMode::MAX}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<ov::op::PhiloxAlignment>& EnumNames<ov::op::PhiloxAlignment>::get() {
    static auto enum_names = EnumNames<ov::op::PhiloxAlignment>("ov::op::PhiloxAlignment",
                                                                {{"pytorch", ov::op::PhiloxAlignment::PYTORCH},
                                                                 {"tensorflow", ov::op::PhiloxAlignment::TENSORFLOW},
                                                                 {"mock", ov::op::PhiloxAlignment::MOCK}});
    return enum_names;
}

bool AttributeAdapter<ov::op::AutoBroadcastSpec>::visit_attributes(AttributeVisitor& visitor) {
    // Maintain back-compatibility
    std::string name = visitor.finish_structure();
    visitor.on_attribute(name, m_ref.m_type);
    visitor.start_structure(name);
    if (m_ref.m_type == ov::op::AutoBroadcastType::PDPD) {
        visitor.on_attribute("auto_broadcast_axis", m_ref.m_axis);
    }
    return true;
}

bool AttributeAdapter<ov::op::BroadcastModeSpec>::visit_attributes(AttributeVisitor& visitor) {
    // Maintain back-compatibility
    std::string name = visitor.finish_structure();
    visitor.on_attribute(name, m_ref.m_type);
    visitor.start_structure(name);
    if (m_ref.m_type == ov::op::BroadcastType::PDPD) {
        visitor.start_structure(name);
        visitor.on_attribute("axis", m_ref.m_axis);
        visitor.finish_structure();
    }
    return true;
}

template <>
OPENVINO_API EnumNames<ov::op::RecurrentSequenceDirection>& EnumNames<ov::op::RecurrentSequenceDirection>::get() {
    static auto enum_names = EnumNames<ov::op::RecurrentSequenceDirection>(
        "ov::op::RecurrentSequenceDirection",
        {{"forward", ov::op::RecurrentSequenceDirection::FORWARD},
         {"reverse", ov::op::RecurrentSequenceDirection::REVERSE},
         {"bidirectional", ov::op::RecurrentSequenceDirection::BIDIRECTIONAL}});
    return enum_names;
}

std::ostream& op::operator<<(std::ostream& s, const ov::op::PadMode& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ov::op::FillMode& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ov::op::PadType& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ov::op::RoundingType& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ov::op::BroadcastType& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ov::op::AutoBroadcastType& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ov::op::EpsMode& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ov::op::TopKSortType& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ov::op::TopKMode& type) {
    return s << as_string(type);
}

std::ostream& op::operator<<(std::ostream& s, const ov::op::PhiloxAlignment& type) {
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

    OPENVINO_ASSERT(allowed_values.count(lowercase_type) > 0, "Invalid 'type' value passed in.");

    return allowed_values.at(lowercase_type);
}

std::ostream& op::operator<<(std::ostream& s, const ov::op::RecurrentSequenceDirection& direction) {
    return s << as_string(direction);
}

AttributeAdapter<op::PadMode>::~AttributeAdapter() = default;
AttributeAdapter<op::FillMode>::~AttributeAdapter() = default;
AttributeAdapter<op::PadType>::~AttributeAdapter() = default;
AttributeAdapter<op::RoundingType>::~AttributeAdapter() = default;
AttributeAdapter<op::AutoBroadcastType>::~AttributeAdapter() = default;
AttributeAdapter<op::BroadcastType>::~AttributeAdapter() = default;
AttributeAdapter<op::EpsMode>::~AttributeAdapter() = default;
AttributeAdapter<op::TopKSortType>::~AttributeAdapter() = default;
AttributeAdapter<op::TopKMode>::~AttributeAdapter() = default;
AttributeAdapter<op::PhiloxAlignment>::~AttributeAdapter() = default;
AttributeAdapter<op::AutoBroadcastSpec>::~AttributeAdapter() = default;
AttributeAdapter<op::BroadcastModeSpec>::~AttributeAdapter() = default;
AttributeAdapter<op::RecurrentSequenceDirection>::~AttributeAdapter() = default;
}  // namespace ov
