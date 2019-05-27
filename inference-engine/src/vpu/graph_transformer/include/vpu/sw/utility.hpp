// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <unordered_set>

#include <ie_parallel.hpp>

#include <vpu/model/data.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

namespace ie = InferenceEngine;

//
// Relayout
//

template <typename T>
void kchw_to_hwck(const T* src, T* dst, const DataDesc& desc) {
    IE_ASSERT(desc.numDims() >= 3);

    auto W = desc.dim(Dim::W);
    auto H = desc.dim(Dim::H);
    auto C = desc.dim(Dim::C);

    ie::parallel_for3d(W, H, C, [=](int w, int h, int c) {
        auto inInd  = w + W * h + W * H * c;
        auto outInd = c + C * h + C * H * w;
        dst[outInd] = src[inInd];
    });
}

template <typename T>
void kchw_to_khwc(const T* src, T* dst, const DataDesc& desc) {
    IE_ASSERT(desc.numDims() >= 3);

    auto W = desc.dim(Dim::W);
    auto H = desc.dim(Dim::H);
    auto C = desc.dim(Dim::C);

    ie::parallel_for3d(W, H, C, [=](int w, int h, int c) {
        auto inInd  = w + W * h + W * H * c;
        auto outInd = h + H * w + H * W * c;
        dst[outInd] = src[inInd];
    });
}

template <typename T>
void kchw_to_hwkc(const T* src, T* dst, const DataDesc& desc) {
    IE_ASSERT(desc.numDims() >= 3);

    auto W = desc.dim(Dim::W);
    auto H = desc.dim(Dim::H);
    auto C = desc.dim(Dim::C);

    ie::parallel_for3d(W, H, C, [=](int w, int h, int c) {
        auto inInd  = w + W * h + W * H * c;
        auto outInd = h + H * c + C * H * w;
        dst[outInd] = src[inInd];
    });
}

template <typename T>
void deconv_to_conv(const T* src, T* dst, const DataDesc& desc) {
    IE_ASSERT(desc.numDims() >= 4);

    auto KX = desc.dim(Dim::W);
    auto KY = desc.dim(Dim::H);
    auto IC = desc.dim(Dim::C);
    auto OC = desc.dim(Dim::N);

    ie::parallel_for4d(OC, IC, KY, KX, [=](int oc, int ic, int ky, int kx) {
        auto inInd = kx + ky * KX + oc * KX * KY + ic * KX * KY * OC;
        auto outInd = (KX - kx - 1) + (KY - ky - 1) * KX + ic * KX * KY + oc * KX * KY * IC;
        dst[outInd] = src[inInd];
    });
}

//
// DefaultSwWeightsContent
//

class DefaultSwWeightsContent final : public CalculatedDataContent {
public:
    explicit DefaultSwWeightsContent(const DataContent::Ptr& origContent);

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override;
};

//
// getNextStage
//

Stage getNextStage(
        const Stage& curStage,
        const std::unordered_set<StageType, EnumClassHash>& supportedTypes);

}  // namespace vpu
