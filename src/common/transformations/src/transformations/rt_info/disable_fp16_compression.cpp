// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/disable_fp16_compression.hpp"

namespace {
const std::string& get_postponed_fp16_compression_tag() {
    static const std::string postponed_fp16_compression_tag("postponed_fp16_compression");
    return postponed_fp16_compression_tag;
}
}  // namespace

void ov::disable_fp16_compression(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[DisableFP16Compression::get_type_info_static()] = DisableFP16Compression{};
}

void ov::enable_fp16_compression(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(DisableFP16Compression::get_type_info_static());
}

bool ov::fp16_compression_is_disabled(const std::shared_ptr<const Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(DisableFP16Compression::get_type_info_static());
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

void ov::disable_compression_to(const std::shared_ptr<Node>& node, element::Type to) {
    return disable_compression_from_to(node, element::dynamic, to);
}

void ov::disable_compression_from_to(const std::shared_ptr<Node>& node, element::Type from, element::Type to) {
    auto& rt_info = node->get_rt_info();
    if (rt_info.count(DisablePrecisionConversion::get_type_info_static())) {
        auto& dpc_attribute =
            rt_info.at(DisablePrecisionConversion::get_type_info_static()).as<DisablePrecisionConversion>();
        dpc_attribute.m_disabled_precisions[from].insert(to);
    } else {
        rt_info[DisablePrecisionConversion::get_type_info_static()] = DisablePrecisionConversion(from, to);
    }
}

void ov::enable_compression_to(const std::shared_ptr<Node>& node, element::Type to) {
    enable_compression_from_to(node, element::dynamic, to);
}

void ov::enable_compression_from_to(const std::shared_ptr<Node>& node, element::Type from, element::Type to) {
    auto& rt_info = node->get_rt_info();
    if (rt_info.count(DisablePrecisionConversion::get_type_info_static())) {
        auto& dpc_attribute =
            rt_info.at(DisablePrecisionConversion::get_type_info_static()).as<DisablePrecisionConversion>();
        if (dpc_attribute.m_disabled_precisions.count(from)) {
            dpc_attribute.m_disabled_precisions[from].erase(to);
            if (dpc_attribute.m_disabled_precisions[from].empty() && from != element::dynamic) {
                dpc_attribute.m_disabled_precisions.erase(from);
            }
        }
    }
}

bool ov::is_compression_disabled_to(const std::shared_ptr<Node>& node, element::Type to) {
    return is_compression_disabled_from_to(node, element::dynamic, to);
}

bool ov::is_compression_disabled_from_to(const std::shared_ptr<Node>& node, element::Type from, element::Type to) {
    auto& rt_info = node->get_rt_info();
    if (rt_info.count(DisablePrecisionConversion::get_type_info_static())) {
        auto& dpc_attribute =
            rt_info.at(DisablePrecisionConversion::get_type_info_static()).as<DisablePrecisionConversion>();

        if (dpc_attribute.m_disabled_precisions.at(element::dynamic).count(element::dynamic) ||
            dpc_attribute.m_disabled_precisions.at(element::dynamic).count(to)) {
            return true;
        }

        if (dpc_attribute.m_disabled_precisions.count(from) && dpc_attribute.m_disabled_precisions.at(from).count(to)) {
            return true;
        }

        return false;
    }
    return false;
}