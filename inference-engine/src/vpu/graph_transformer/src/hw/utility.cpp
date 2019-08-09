// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/hw/utility.hpp>

#include <string>
#include <unordered_map>
#include <algorithm>

#include <ie_parallel.hpp>

#include <vpu/model/stage.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/utils/profiling.hpp>

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

//
// HwWeightsContent
//

HwWeightsContent::HwWeightsContent(const DataContent::Ptr& origContent,
        const DataDesc& origWeightsDesc,
        int numInputChannels,
        int channelStartIndex) :
        CalculatedDataContent({origContent}),
        _origWeightsDesc(origWeightsDesc),
        _numInputChannels(numInputChannels),
        _channelStartIndex(channelStartIndex) {
}

void HwWeightsContent::fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const {
    VPU_PROFILE(HwWeightsContent);

    IE_ASSERT(_desc.type() == DataType::FP16);
    IE_ASSERT(baseContents.size() == 1);

    auto KX = _origWeightsDesc.dim(Dim::W);
    auto KY = _origWeightsDesc.dim(Dim::H);
    auto IC = _origWeightsDesc.dim(Dim::C);
    auto OC = _origWeightsDesc.dim(Dim::N);
    auto origTotalSize = _origWeightsDesc.totalDimSize();

    auto HW_OC_inner = desc().dim(Dim::W);
    auto HW_OC_outer = desc().dim(Dim::N);
    IE_ASSERT(HW_OC_outer * HW_OC_inner >= OC);

    auto HW_K = desc().dim(Dim::H);
    IE_ASSERT(HW_K == KX * KY);

    IE_ASSERT(_channelStartIndex < IC);
    auto HW_IC = desc().dim(Dim::C);
    auto HW_IC_real = std::min(_numInputChannels, IC - _channelStartIndex);

    auto srcData = baseContents[0]->get<fp16_t>();
    IE_ASSERT(srcData != nullptr);

    auto dstData = static_cast<fp16_t*>(tempBuf);

    IE_ASSERT((_channelStartIndex + HW_IC_real) * HW_K + (OC - 1) * HW_K * IC - 1 < origTotalSize);
    IE_ASSERT((OC - 1) % HW_OC_inner +
              (HW_K - 1) * HW_OC_inner +
              (HW_IC_real - 1) * HW_OC_inner * HW_K +
              ((OC - 1) / 8) * HW_OC_inner * HW_K * HW_IC < _desc.totalDimSize());

    if (KX == 1 && KY == 1) {
        ie::parallel_for(OC, [=](int oc) {
            auto oc_inner = oc % HW_OC_inner;
            auto oc_outer = oc / HW_OC_inner;
            for (int ic = 0; ic < HW_IC_real; ++ic) {
                auto srcInd =
                        (_channelStartIndex + ic) +
                        oc * IC;
                auto dstInd =
                        oc_inner +
                        ic * HW_OC_inner * HW_K +
                        oc_outer * HW_OC_inner * HW_K * HW_IC;

                dstData[dstInd] = srcData[srcInd];
            }
        });
    } else {
        ie::parallel_for(OC, [=](int oc) {
            auto oc_inner = oc % HW_OC_inner;
            auto oc_outer = oc / HW_OC_inner;
            for (int ic = 0; ic < HW_IC_real; ++ic) {
                for (int ky = 0; ky < KY; ++ky) {
                    for (int kx = 0; kx < KX; ++kx) {
                        auto srcInd =
                                (kx + ky * KX) +
                                (_channelStartIndex + ic) * HW_K +
                                oc * HW_K * IC;
                        auto dstInd =
                                oc_inner +
                                (ky * KX + kx) * HW_OC_inner +
                                ic * HW_OC_inner * HW_K +
                                oc_outer * HW_OC_inner * HW_K * HW_IC;

                        dstData[dstInd] = srcData[srcInd];
                    }
                }
            }
        });
    }
}
//
// calculateHwBufferSize
//

int calculateHwBufferSize(const DimValues& dims, DimsOrder order) {
    if (order.empty()) {
        order = DimsOrder::fromNumDims(dims.size());
    }

    DataDesc desc(DataType::FP16, order, dims);

    if (desc.numDims() > 2) {
        return calcTotalByteSize(desc, calcStrides(desc, StridesRequirement().add(1, DimStride::Aligned)));
    } else {
        IE_ASSERT(desc.dimsOrder() == DimsOrder::NC);

        return calcTotalByteSize(desc, calcStrides(desc, StridesRequirement().add(0, DimStride::Aligned)));
    }
}

}  // namespace vpu
