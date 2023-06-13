// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
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

inline std::string getImplPriorityValue(const std::shared_ptr<ngraph::Node> &node) {
    const auto &rtInfo = node->get_rt_info();

    auto it_info = rtInfo.find(ov::PrimitivesPriority::get_type_info_static());

    if (it_info == rtInfo.end()) {
        return {};
    }

    return it_info->second.as<ov::PrimitivesPriority>().value;
}

template <typename T>
inline const std::shared_ptr<T> getNgraphOpAs(const std::shared_ptr<ngraph::Node>& op) {
    auto typedOp = ngraph::as_type_ptr<T>(op);
    if (!typedOp)
        IE_THROW() << "Can't get ngraph node " << op->get_type_name() << " with name " << op->get_friendly_name();
    return typedOp;
}

inline bool isDynamicNgraphNode(const std::shared_ptr<const ngraph::Node>& op) {
    bool ret = op->is_dynamic();
    for (size_t i = 0; i < op->get_output_size(); i++) {
        ret = ret || op->get_output_partial_shape(i).is_dynamic();
    }
    return ret;
}

}   // namespace intel_cpu
}   // namespace ov
