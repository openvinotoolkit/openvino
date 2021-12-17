// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <ostream>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include "mask_attribute.hpp"

namespace ngraph {

static const char g_init_mask_key[] = "InitMask";
static const char g_result_pruning_mask_key[] = "ResultMask";

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

void setResultMask(Output<Node> output, const Mask::Ptr & mask) {
    auto &rtInfo = output.get_rt_info();
    rtInfo[g_result_pruning_mask_key] = mask;
}

Mask::Ptr getInitMask(const Output<Node> & output) {
    auto &rtInfo = output.get_rt_info();

    const auto attr_it = rtInfo.find(g_init_mask_key);
    if (attr_it == rtInfo.end()) return nullptr;

    const auto &attr = attr_it->second;
    return attr.as<Mask::Ptr>();
}

void setInitMask(Output<Node> output, const Mask::Ptr & mask) {
    auto &rtInfo = output.get_rt_info();
    auto copy_mask = std::make_shared<Mask>();
    std::copy(mask->begin(), mask->end(), std::back_inserter(*copy_mask));
    rtInfo[g_init_mask_key] = copy_mask;
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
