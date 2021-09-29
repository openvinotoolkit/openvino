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
    using MaskWrapper = VariantWrapper<Mask::Ptr>;

    if (!rtInfo.count(MaskWrapper::get_type_info_static().name)) return nullptr;

    const auto &attr = rtInfo.at(MaskWrapper::get_type_info_static().name);
    return ov::as_type_ptr<MaskWrapper>(attr)->get();
}

Mask::Ptr getMask(const Output<Node> & output) {
    auto &rtInfo = output.get_rt_info();
    using MaskWrapper = VariantWrapper<Mask::Ptr>;

    if (!rtInfo.count(MaskWrapper::get_type_info_static().name)) return nullptr;

    const auto &attr = rtInfo.at(MaskWrapper::get_type_info_static().name);
    return ov::as_type_ptr<MaskWrapper>(attr)->get();
}

void setMask(Output<Node> output, const Mask::Ptr & mask) {
    auto &rtInfo = output.get_rt_info();
    using MaskWrapper = VariantWrapper<Mask::Ptr>;
    rtInfo[MaskWrapper::get_type_info_static().name] = MaskWrapper::create(mask);
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

namespace ov {

template class ngraph::VariantImpl<ngraph::Mask::Ptr>;

BWDCMP_RTTI_DEFINITION(VariantWrapper<ngraph::Mask::Ptr>);

}  // namespace ov
