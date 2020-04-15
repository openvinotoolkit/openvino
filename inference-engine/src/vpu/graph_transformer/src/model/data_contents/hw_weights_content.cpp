// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/hw_weights_content.hpp>

#include <vpu/utils/profiling.hpp>

#include <ie_parallel.hpp>

namespace vpu {

HwWeightsContent::HwWeightsContent(const DataContent::Ptr& origContent,
                                   const DataDesc& origWeightsDesc,
                                   const DataDesc& resDesc,
                                   int numInputChannels,
                                   int channelStartIndex) :
        _origContent(origContent),
        _origDesc(origWeightsDesc),
        _resDesc(resDesc),
        _numInputChannels(numInputChannels),
        _channelStartIndex(channelStartIndex) {
}

size_t HwWeightsContent::byteSize() const {
    return checked_cast<size_t>(_resDesc.totalDimSize()) *
           checked_cast<size_t>(_resDesc.elemSize());
}

void HwWeightsContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(HwWeightsContent);

    IE_ASSERT(_resDesc.type() == DataType::FP16);

    const auto KX = _origDesc.dim(Dim::W);
    const auto KY = _origDesc.dim(Dim::H);
    const auto IC = _origDesc.dim(Dim::C);
    const auto OC = _origDesc.dim(Dim::N);
    const auto origTotalSize = _origDesc.totalDimSize();

    const auto HW_OC_inner = _resDesc.dim(Dim::W);
    const auto HW_OC_outer = _resDesc.dim(Dim::N);
    IE_ASSERT(HW_OC_outer * HW_OC_inner >= OC);

    const auto HW_K = _resDesc.dim(Dim::H);
    IE_ASSERT(HW_K == KX * KY);

    IE_ASSERT(_channelStartIndex < IC);
    const auto HW_IC = _resDesc.dim(Dim::C);
    const auto HW_IC_real = std::min(_numInputChannels, IC - _channelStartIndex);

    const auto srcData = _origContent->get<fp16_t>();
    IE_ASSERT(srcData != nullptr);

    auto dstData = static_cast<fp16_t*>(tempBuf);

    IE_ASSERT((_channelStartIndex + HW_IC_real) * HW_K + (OC - 1) * HW_K * IC - 1 < origTotalSize);
    IE_ASSERT((OC - 1) % HW_OC_inner +
              (HW_K - 1) * HW_OC_inner +
              (HW_IC_real - 1) * HW_OC_inner * HW_K +
              ((OC - 1) / 8) * HW_OC_inner * HW_K * HW_IC < _resDesc.totalDimSize());

    if (KX == 1 && KY == 1) {
        ie::parallel_for(OC, [=](int oc) {
            const auto oc_inner = oc % HW_OC_inner;
            const auto oc_outer = oc / HW_OC_inner;
            for (int ic = 0; ic < HW_IC_real; ++ic) {
                const auto srcInd =
                        (_channelStartIndex + ic) +
                        oc * IC;
                const auto dstInd =
                        oc_inner +
                        ic * HW_OC_inner * HW_K +
                        oc_outer * HW_OC_inner * HW_K * HW_IC;

                dstData[dstInd] = srcData[srcInd];
            }
        });
    } else {
        ie::parallel_for(OC, [=](int oc) {
            const auto oc_inner = oc % HW_OC_inner;
            const auto oc_outer = oc / HW_OC_inner;
            for (int ic = 0; ic < HW_IC_real; ++ic) {
                for (int ky = 0; ky < KY; ++ky) {
                    for (int kx = 0; kx < KX; ++kx) {
                        const auto srcInd =
                                (kx + ky * KX) +
                                (_channelStartIndex + ic) * HW_K +
                                oc * HW_K * IC;
                        const auto dstInd =
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

} // namespace vpu
