// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/disable_precision_conversion.hpp"

#include <sstream>

#include "openvino/util/common_util.hpp"

namespace {
const std::string& get_postponed_fp16_compression_tag() {
    static const std::string postponed_fp16_compression_tag("postponed_fp16_compression");
    return postponed_fp16_compression_tag;
}
}  // namespace

void ov::postpone_fp16_compression(ov::RTMap& rt_info) {
    rt_info[get_postponed_fp16_compression_tag()] = true;
}

bool ov::is_fp16_compression_postponed(const ov::RTMap& rt_info) {
    return rt_info.count(get_postponed_fp16_compression_tag());
}

void ov::do_not_postpone_fp16_compression(ov::RTMap& rt_info) {
    rt_info.erase(get_postponed_fp16_compression_tag());
}

void ov::disable_conversion(const std::shared_ptr<Node>& node, const element::Type& to) {
    return disable_conversion(node, element::dynamic, to);
}

void ov::disable_conversion(const std::shared_ptr<Node>& node, const element::Type& from, const element::Type& to) {
    auto& rt_info = node->get_rt_info();
    auto it = rt_info.find(DisablePrecisionConversion::get_type_info_static());
    if (it != rt_info.end()) {
        auto& dpc_attribute = it->second.as<DisablePrecisionConversion>();
        dpc_attribute.m_disabled_precisions[from].insert(to);
    } else {
        rt_info[DisablePrecisionConversion::get_type_info_static()] = DisablePrecisionConversion(from, to);
    }
}

void ov::disable_conversion(const std::shared_ptr<Node>& node,
                            const element::TypeVector& from_types,
                            const element::TypeVector& to_types) {
    for (const auto& from : from_types) {
        for (const auto& to : to_types) {
            disable_conversion(node, from, to);
        }
    }
}

void ov::enable_conversion(const std::shared_ptr<Node>& node, const element::Type& to) {
    enable_conversion(node, element::dynamic, to);
}

void ov::enable_conversion(const std::shared_ptr<Node>& node, const element::Type& from, const element::Type& to) {
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

void ov::enable_conversion(const std::shared_ptr<Node>& node,
                           const element::TypeVector& from_types,
                           const element::TypeVector& to_types) {
    for (const auto& from : from_types) {
        for (const auto& to : to_types) {
            enable_conversion(node, from, to);
        }
    }
}

bool ov::is_conversion_disabled(const std::shared_ptr<const Node>& node, const element::Type& to) {
    return is_conversion_disabled(node, element::dynamic, to);
}

bool ov::is_conversion_disabled(const std::shared_ptr<const Node>& node,
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

// Format: "from:to1,to2;from2:to3,to4" - entries separated by ';' (no ';' at the end)
const std::string& ov::AttributeAdapter<ov::DisabledPrecisionMap>::get() {
    std::ostringstream oss;
    auto it = m_ref.begin();
    if (it != m_ref.end()) {
        oss << it->first << ':' << ov::util::join<std::ostream>(it->second, ",");
        while (++it != m_ref.end()) {
            const auto& [from, to] = *it;
            oss << ';' << from << ':' << ov::util::join<std::ostream>(to, ",");
        }
    }
    m_serialized = oss.str();
    return m_serialized;
}

void ov::AttributeAdapter<ov::DisabledPrecisionMap>::set(const std::string& value) {
    m_ref.clear();
    for (std::string_view sv(value); !sv.empty();) {
        const auto sep_pos = sv.find(';');
        const auto entry = sv.substr(0, sep_pos);
        sv = sep_pos != std::string_view::npos ? sv.substr(sep_pos + 1) : std::string_view{};
        if (const auto colon_pos = entry.find(':'); colon_pos != std::string_view::npos) {
            element::Type from_type(std::string(entry.substr(0, colon_pos)));
            auto& to_set = m_ref[from_type];
            ov::util::view_transform(entry.substr(colon_pos + 1), std::inserter(to_set, to_set.end()), ",", [](auto s) {
                return element::Type(std::string(s));
            });
        }
    }
}