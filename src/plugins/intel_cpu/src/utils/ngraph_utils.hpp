// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include "ie_common.h"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"

namespace ov {
namespace intel_cpu {

inline std::string getRTInfoValue(const ov::Node::RTMap& rtInfo, std::string paramName) {
    auto it = rtInfo.find(paramName);
    if (it != rtInfo.end()) {
        return it->second.as<std::string>();
    } else {
        return {};
    }
}

inline std::string getPrimitivesPriorityValue(const ov::Node::RTMap& rtInfo) {
    auto it_info = rtInfo.find(ov::PrimitivesPriority::get_type_info_static());

    if (it_info == rtInfo.end()) {
        return {};
    }

    return it_info->second.as<ov::PrimitivesPriority>().value;
}

inline std::string getInputMemoryFormats(const ov::Node::RTMap& rtInfo) {
    auto it_info = rtInfo.find(InputMemoryFormats::get_type_info_static());

    if (it_info == rtInfo.end()) {
        return {};
    }

    return it_info->second.as<InputMemoryFormats>().to_string();
}

inline std::string getOutputMemoryFormats(const ov::Node::RTMap& rtInfo) {
    auto it_info = rtInfo.find(OutputMemoryFormats::get_type_info_static());

    if (it_info == rtInfo.end()) {
        return {};
    }

    return it_info->second.as<OutputMemoryFormats>().to_string();
}

template <typename T>
inline const std::shared_ptr<T> getNgraphOpAs(const std::shared_ptr<ngraph::Node>& op) {
    auto typedOp = ngraph::as_type_ptr<T>(op);
    if (!typedOp)
        IE_THROW() << "Can't get ngraph node " << op->get_type_name() << " with name " << op->get_friendly_name();
    return typedOp;
}

inline bool isDynamicNgraphNode(const std::shared_ptr<const ngraph::Node>& op) {
    if (op->is_dynamic())
        return true;

    for (size_t i = 0; i < op->get_output_size(); i++) {
        if (op->get_output_partial_shape(i).is_dynamic())
            return true;
    }

    return false;
}

}   // namespace intel_cpu
}   // namespace ov
