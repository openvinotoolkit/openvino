// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <ngraph/variant.hpp>
#include "transformations/rt_info/primitives_priority_attribute.hpp"

namespace MKLDNNPlugin {

inline std::string getRTInfoValue(const std::map<std::string, std::shared_ptr<ngraph::Variant>>& rtInfo, std::string paramName) {
    auto it = rtInfo.find(paramName);
    if (it != rtInfo.end()) {
        auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
        return value->get();
    } else {
        return "";
    }
}

inline std::string getPrimitivesPriorityValue(const std::shared_ptr<ngraph::Node> &node) {
    const auto &rtInfo = node->get_rt_info();

    if (!rtInfo.count(ov::PrimitivesPriority::get_type_info_static())) return "";

    const auto &attr = rtInfo.at(ov::PrimitivesPriority::get_type_info_static());
    return ngraph::as_type_ptr<ov::PrimitivesPriority>(attr)->get();
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

}  // namespace MKLDNNPlugin
