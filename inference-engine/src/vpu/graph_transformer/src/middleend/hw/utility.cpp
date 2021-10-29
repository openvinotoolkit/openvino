// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/hw/utility.hpp>

#include <string>
#include <unordered_map>
#include <algorithm>

#include <vpu/model/stage.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/middleend/allocator/structs.hpp>

namespace vpu {

//
// HwDescriptors
//

void printTo(std::ostream& os, const HwOpList& hwOps) {
    os << "[" << std::endl;
    os << "size=" << hwOps.vec.size() << std::endl;
    os << "]";
}

void printTo(DotLabel& lbl, const HwOpList& hwOps) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("size", hwOps.vec.size());
}

//
// HwPaddingInfo
//

HwPaddingInfo getHwPaddingInfo(
        const DimValues& inDims, const DimValues& outDims,
        int kernelDimX, int kernelDimY,
        int kernelStrideX, int kernelStrideY,
        int padLeft, int padTop) {
    auto pad_along_x = (outDims[Dim::W] - 1) * kernelStrideX + kernelDimX - inDims[Dim::W];
    auto pad_along_y = (outDims[Dim::H] - 1) * kernelStrideY + kernelDimY - inDims[Dim::H];

    HwPaddingInfo pad;

    pad.left   = padLeft;
    pad.right  = std::max(0, pad_along_x - pad.left);
    pad.top    = padTop;
    pad.bottom = std::max(0, pad_along_y - pad.top);

    pad.enable = pad.left || pad.right || pad.top || pad.bottom;

    return pad;
}

void printTo(std::ostream& os, const HwPaddingInfo& hwPad) {
    os << "[" << std::endl;
    os << "enable=" << hwPad.enable << std::endl;
    if (hwPad.enable) {
        os << "left=" << hwPad.left << std::endl;
        os << "right=" << hwPad.right << std::endl;
        os << "top=" << hwPad.top << std::endl;
        os << "bottom=" << hwPad.bottom << std::endl;
    }
    os << "]";
}

void printTo(DotLabel& lbl, const HwPaddingInfo& hwPad) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("enable", hwPad.enable);
    if (hwPad.enable) {
        subLbl.appendPair("left", hwPad.left);
        subLbl.appendPair("right", hwPad.right);
        subLbl.appendPair("top", hwPad.top);
        subLbl.appendPair("bottom", hwPad.bottom);
    }
}

int calculateHwBufferSize(const DimValues& dims, const DimsOrder& order) {
    const auto desc = DataDesc{DataType::FP16, order.empty() ? DimsOrder::fromNumDims(dims.size()) : order, dims};
    IE_ASSERT(desc.numDims() > 2 || desc.dimsOrder() == DimsOrder::NC);

    const auto channelIndex = desc.numDims() > 2 ? 1 : 0;
    const auto strides = calcStrides(desc, StridesRequirement().add(channelIndex, DimStride::Aligned));
    return calcTotalByteSize(desc, strides);
}

}  // namespace vpu
