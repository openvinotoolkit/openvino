// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <transformations/utils/utils.hpp>

#include "transformations/rt_info/primitives_priority_attribute.hpp"

namespace ov {
namespace intel_cpu {

inline std::string getRTInfoValue(const std::map<std::string, ov::Any>& rtInfo, std::string paramName) {
    auto it = rtInfo.find(paramName);
    if (it != rtInfo.end()) {
        return it->second.as<std::string>();
    } else {
        return {};
    }
}

inline std::string getImplPriorityValue(const std::shared_ptr<ov::Node> &node) {
    const auto &rtInfo = node->get_rt_info();

    auto it_info = rtInfo.find(ov::PrimitivesPriority::get_type_info_static());

    if (it_info == rtInfo.end()) {
        return {};
    }

    return it_info->second.as<ov::PrimitivesPriority>().value;
}

template <typename T>
inline const std::shared_ptr<T> getNgraphOpAs(const std::shared_ptr<ov::Node>& op) {
    auto typedOp = ov::as_type_ptr<T>(op);
    if (!typedOp)
        OPENVINO_THROW("Can't get ngraph node ", op->get_type_name(), " with name ", op->get_friendly_name());
    return typedOp;
}

inline bool isDynamicNgraphNode(const std::shared_ptr<const ov::Node>& op) {
    bool ret = op->is_dynamic();
    for (size_t i = 0; i < op->get_output_size(); i++) {
        ret = ret || op->get_output_partial_shape(i).is_dynamic();
    }
    return ret;
}

inline std::string get_port_name(const ov::Output<const ov::Node>& port, const bool is_legacy_api) {
    std::string name;
    // Should use tensor name as the port name, but many legacy tests still use legacy name
    // plus sometimes it will get empty tensor name.
    if (!is_legacy_api) {
        // TODO: To apply unified tensor name.
    }
    if (name.empty()) {
        bool is_input = ov::op::util::is_parameter(port.get_node());
        if (is_input) {
            name = ov::op::util::get_ie_output_name(port);
        } else {
            const auto node = port.get_node_shared_ptr();
            name = ov::op::util::get_ie_output_name(node->input_value(0));
        }
    }
    return name;
}

}   // namespace intel_cpu
}   // namespace ov
