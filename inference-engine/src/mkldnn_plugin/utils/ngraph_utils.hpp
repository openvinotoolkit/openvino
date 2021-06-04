// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <cassert>
#include <algorithm>
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
    using PrimitivesPriorityWraper = ngraph::VariantWrapper<ngraph::PrimitivesPriority>;

    if (!rtInfo.count(PrimitivesPriorityWraper::type_info.name)) return "";

    const auto &attr = rtInfo.at(PrimitivesPriorityWraper::type_info.name);
    ngraph::PrimitivesPriority pp = ngraph::as_type_ptr<PrimitivesPriorityWraper>(attr)->get();
    return pp.getPrimitivesPriority();
}

template <typename T>
inline const std::shared_ptr<T> getNgraphOpAs(const std::shared_ptr<ngraph::Node>& op) {
    auto typedOp = ngraph::as_type_ptr<T>(op);
    if (!typedOp)
        IE_THROW() << "Can't get ngraph node " << op->get_type_name() << " with name " << op->get_friendly_name();
    return typedOp;
}

template <typename V>
inline typename std::map<const ngraph::Node::type_info_t, V>::const_iterator
find_castable_type_info(const std::map<const ngraph::Node::type_info_t, V>& map,
                        const ngraph::Node::type_info_t& typeInfo) {
    return std::find_if(map.begin(), map.end(),
            [typeInfo](const typename std::map<const ngraph::Node::type_info_t, V>::value_type & n) {
                return typeInfo.is_castable(n.first);
            });
}

}  // namespace MKLDNNPlugin