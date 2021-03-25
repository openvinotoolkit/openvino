// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <ostream>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include "mask_attribute.hpp"

namespace ngraph {

template class ngraph::VariantImpl<Mask::Ptr>;

constexpr VariantTypeInfo VariantWrapper<Mask::Ptr>::type_info;

Mask::Ptr getMask(const Output<const Node> & output) {
    auto &rtInfo = output.get_rt_info();
    using MaskWraper = VariantWrapper<Mask::Ptr>;

    if (!rtInfo.count(MaskWraper::type_info.name)) return nullptr;

    const auto &attr = rtInfo.at(MaskWraper::type_info.name);
    return as_type_ptr<MaskWraper>(attr)->get();
}

Mask::Ptr getMask(const Output<Node> & output) {
    auto &rtInfo = output.get_rt_info();
    using MaskWraper = VariantWrapper<Mask::Ptr>;

    if (!rtInfo.count(MaskWraper::type_info.name)) return nullptr;

    const auto &attr = rtInfo.at(MaskWraper::type_info.name);
    return as_type_ptr<MaskWraper>(attr)->get();
}

void setMask(Output<Node> output, const Mask::Ptr & mask) {
    auto &rtInfo = output.get_rt_info();
    using MaskWraper = VariantWrapper<Mask::Ptr>;
    rtInfo[MaskWraper::type_info.name] = MaskWraper::create(mask);
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
