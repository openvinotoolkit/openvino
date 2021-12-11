// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <ostream>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include "mask_attribute.hpp"

namespace ngraph {

Mask::Ptr getMask(const Output<const Node> & output) {
    auto &rtInfo = output.get_rt_info();
    if (!rtInfo.count(Mask::get_type_info_static())) return nullptr;

    const auto &attr = rtInfo.at(Mask::get_type_info_static());
    return attr.as<Mask::Ptr>();
}

Mask::Ptr getMask(const Output<Node> & output) {
    auto &rtInfo = output.get_rt_info();
    if (!rtInfo.count(Mask::get_type_info_static())) return nullptr;
    const auto &attr = rtInfo.at(Mask::get_type_info_static());
    return attr.as<Mask::Ptr>();
}

void setMask(Output<Node> output, const Mask::Ptr & mask) {
    auto &rtInfo = output.get_rt_info();
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
