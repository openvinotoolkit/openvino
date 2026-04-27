// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/disable_fp16_compression.hpp"

#include <sstream>

#include "openvino/util/common_util.hpp"

namespace {
const std::string& get_postponed_fp16_compression_tag() {
    static const std::string postponed_fp16_compression_tag("postponed_fp16_compression");
    return postponed_fp16_compression_tag;
}
}  // namespace

void ov::disable_fp16_compression(const std::shared_ptr<Node>& node) {
    disable_compression_to(node, element::f16);
}

void ov::enable_fp16_compression(const std::shared_ptr<Node>& node) {
    enable_compression_to(node, element::f16);
}

bool ov::fp16_compression_is_disabled(const std::shared_ptr<const Node>& node) {
    return is_compression_disabled_to(node, element::f16);
}

void ov::postpone_fp16_compression(ov::RTMap& rt_info) {
    rt_info[get_postponed_fp16_compression_tag()] = true;
}

bool ov::is_fp16_compression_postponed(const ov::RTMap& rt_info) {
    return rt_info.count(get_postponed_fp16_compression_tag());
}

void ov::do_not_postpone_fp16_compression(ov::RTMap& rt_info) {
    rt_info.erase(get_postponed_fp16_compression_tag());
}

void ov::disable_compression_to(const std::shared_ptr<Node>& node, const element::Type& to) {
    return disable_compression_from_to(node, element::dynamic, to);
}

void ov::disable_compression_from_to(const std::shared_ptr<Node>& node,
                                     const element::Type& from,
                                     const element::Type& to) {
    auto& rt_info = node->get_rt_info();
    auto it = rt_info.find(DisablePrecisionConversion::get_type_info_static());
    if (it != rt_info.end()) {
        auto& dpc_attribute = it->second.as<DisablePrecisionConversion>();
        dpc_attribute.m_disabled_precisions[from].insert(to);
    } else {
        rt_info[DisablePrecisionConversion::get_type_info_static()] = DisablePrecisionConversion(from, to);
    }
}

void ov::disable_compression_from_to(const std::shared_ptr<Node>& node,
                                     const std::vector<element::Type>& from_types,
                                     const std::vector<element::Type>& to_types) {
    for (const auto& from : from_types) {
        for (const auto& to : to_types) {
            disable_compression_from_to(node, from, to);
        }
    }
}

void ov::enable_compression_to(const std::shared_ptr<Node>& node, const element::Type& to) {
    enable_compression_from_to(node, element::dynamic, to);
}

void ov::enable_compression_from_to(const std::shared_ptr<Node>& node,
                                    const element::Type& from,
                                    const element::Type& to) {
    auto& rt_info = node->get_rt_info();
    auto it = rt_info.find(DisablePrecisionConversion::get_type_info_static());
    if (it != rt_info.end()) {
        auto& dpc_attribute = it->second.as<DisablePrecisionConversion>();
        auto from_it = dpc_attribute.m_disabled_precisions.find(from);
        if (from_it != dpc_attribute.m_disabled_precisions.end()) {
            from_it->second.erase(to);
            if (from_it->second.empty()) {
                dpc_attribute.m_disabled_precisions.erase(from_it);
            }
        }
    }
}

void ov::enable_compression_from_to(const std::shared_ptr<Node>& node,
                                    const std::vector<element::Type>& from_types,
                                    const std::vector<element::Type>& to_types) {
    for (const auto& from : from_types) {
        for (const auto& to : to_types) {
            enable_compression_from_to(node, from, to);
        }
    }
}

bool ov::is_compression_disabled_to(const std::shared_ptr<const Node>& node, const element::Type& to) {
    return is_compression_disabled_from_to(node, element::dynamic, to);
}

bool ov::is_compression_disabled_from_to(const std::shared_ptr<const Node>& node,
                                         const element::Type& from,
                                         const element::Type& to) {
    auto& rt_info = node->get_rt_info();
    auto it = rt_info.find(DisablePrecisionConversion::get_type_info_static());
    if (it != rt_info.end()) {
        auto& dpc_attribute = it->second.as<DisablePrecisionConversion>();

        auto dyn_it = dpc_attribute.m_disabled_precisions.find(element::dynamic);
        if (dyn_it != dpc_attribute.m_disabled_precisions.end() &&
            (dyn_it->second.count(element::dynamic) || dyn_it->second.count(to))) {
            return true;
        }

        auto from_it = dpc_attribute.m_disabled_precisions.find(from);
        if (from_it != dpc_attribute.m_disabled_precisions.end()) {
            const auto& to_set = from_it->second;
            if (to_set.count(to) || to_set.count(element::dynamic)) {
                return true;
            }
        }

        return false;
    }
    return false;
}

bool ov::DisablePrecisionConversion::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("value", m_disabled_precisions);
    return true;
}

const std::string& ov::AttributeAdapter<ov::DisabledPrecisionMap>::get() {
    std::ostringstream oss;
    bool first_entry = true;
    for (const auto& [from_type, to_types] : m_ref) {
        if (!first_entry)
            oss << ';';
        first_entry = false;
        oss << from_type.to_string() << ':' << ov::util::join(to_types, ",");
    }
    m_serialized = oss.str();
    return m_serialized;
}

void ov::AttributeAdapter<ov::DisabledPrecisionMap>::set(const std::string& value) {
    m_ref.clear();
    if (value.empty())
        return;
    std::istringstream iss(value);
    std::string entry;
    while (std::getline(iss, entry, ';')) {
        auto colon_pos = entry.find(':');
        if (colon_pos == std::string::npos)
            continue;
        element::Type from_type(entry.substr(0, colon_pos));
        std::string to_part = entry.substr(colon_pos + 1);
        auto& to_set = m_ref[from_type];
        if (!to_part.empty()) {
            std::istringstream to_stream(to_part);
            std::string to_name;
            while (std::getline(to_stream, to_name, ',')) {
                to_set.insert(element::Type(to_name));
            }
        }
    }
}