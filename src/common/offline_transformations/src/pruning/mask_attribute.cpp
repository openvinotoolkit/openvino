// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <ostream>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include "mask_attribute.hpp"

namespace ngraph {

static const std::string g_init_suffix = "init";

Mask::Ptr getMask(const Output<const Node> & output) {
    auto &rtInfo = output.get_rt_info();

    const auto attr_it = rtInfo.find(Mask::get_type_info_static());
    if (attr_it == rtInfo.end()) return nullptr;

    const auto &attr = attr_it->second;
    return attr.as<Mask::Ptr>();
}

Mask::Ptr getMask(const Output<Node> & output) {
    auto &rtInfo = output.get_rt_info();

    const auto attr_it = rtInfo.find(Mask::get_type_info_static());
    if (attr_it == rtInfo.end()) return nullptr;

    const auto &attr = attr_it->second;
    return attr.as<Mask::Ptr>();
}

void setMask(Output<Node> output, const Mask::Ptr & mask) {
    auto &rtInfo = output.get_rt_info();
    rtInfo[Mask::get_type_info_static()] = mask;
}

Mask::Ptr getInitMask(const Output<Node> & output) {
    auto &rtInfo = output.get_rt_info();
   
    auto init_mask_name = g_init_suffix + Mask::get_type_info_static().name; 

    const auto attr_it = rtInfo.find(init_mask_name);
    if (attr_it == rtInfo.end()) return nullptr;

    const auto &attr = attr_it->second;
    return attr.as<Mask::Ptr>();
}

void setInitMask(Output<Node> output, const Mask::Ptr & mask) {
    auto &rtInfo = output.get_rt_info();

    auto copy_mask = std::make_shared<Mask>();
    std::copy(mask->begin(), mask->end(), std::back_inserter(*copy_mask));
    auto new_name = g_init_suffix + Mask::get_type_info_static().name;
   rtInfo[Mask::get_type_info_static()] = mask;
}

std::ostream & operator<< (std::ostream & out, const Mask & mask) {
    out << "[ ";
    for (auto & dim : mask) {
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

}  // namespace ngraph
