// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include <functional>
#include <memory>
#include <iterator>
#include <ostream>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "transformations/rt_info/mask_attribute.hpp"
#include "ngraph_ops/convolution_ie.hpp"
#include "ngraph_ops/deconvolution_ie.hpp"

namespace ngraph {

template class ngraph::VariantImpl<Mask::Ptr>;

constexpr VariantTypeInfo VariantWrapper<Mask::Ptr>::type_info;

Mask::Ptr getMask(Output<Node> output) {
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
    out << "Mask [ ";
        for (auto & dim : mask) {
            out << "{";
            for (auto & value : *dim) {
                out << value << " ";
            }
        out << "} ";
    }
    out << " ]";
    return out;
}



}  // namespace ngraph
