// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mask_attribute.hpp"

#include <functional>
#include <ostream>

#include "openvino/core/node.hpp"

namespace ov {

Mask::Ptr getMask(const Output<const Node>& output) {
    auto& rtInfo = output.get_rt_info();

    const auto attr_it = rtInfo.find(Mask::get_type_info_static());
    if (attr_it == rtInfo.end())
        return nullptr;

    const auto& attr = attr_it->second;
    return attr.as<Mask::Ptr>();
}

Mask::Ptr getMask(const Output<Node>& output) {
    auto& rtInfo = output.get_rt_info();

    const auto attr_it = rtInfo.find(Mask::get_type_info_static());
    if (attr_it == rtInfo.end())
        return nullptr;

    const auto& attr = attr_it->second;
    return attr.as<Mask::Ptr>();
}

void setMask(Output<Node> output, const Mask::Ptr& mask) {
    auto& rtInfo = output.get_rt_info();
    rtInfo[Mask::get_type_info_static()] = mask;
}

void setMask(Input<Node> node, const Mask::Ptr& mask) {
    auto& rtInfo = node.get_rt_info();
    rtInfo[Mask::get_type_info_static()] = mask;
}

#ifdef ENABLE_OPENVINO_DEBUG
static const char g_init_mask_key[] = "InitMask";
Mask::Ptr getInitMask(const Output<Node>& output) {
    auto& rtInfo = output.get_rt_info();

    const auto attr_it = rtInfo.find(g_init_mask_key);
    if (attr_it == rtInfo.end())
        return nullptr;

    const auto& attr = attr_it->second;
    return attr.as<Mask::Ptr>();
}

void setInitMask(Output<Node> output, const Mask::Ptr& mask) {
    auto& rtInfo = output.get_rt_info();
    auto copy_mask = std::make_shared<Mask>();
    std::copy(mask->begin(), mask->end(), std::back_inserter(*copy_mask));
    rtInfo[g_init_mask_key] = copy_mask;
}
#endif

std::ostream& operator<<(std::ostream& out, const Mask& mask) {
    out << "[ ";
    for (auto& dim : mask) {
        out << "{";
        out << dim.size();
        // Uncomment this to print values
        // for (auto & value : dim) {
        //     out << value << " ";
        // }
        out << "} ";
    }
    out << " ]";
    return out;
}

}  // namespace ov
