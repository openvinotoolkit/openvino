// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/merge_fc_content.hpp>

#include <ie_parallel.hpp>

#include <numeric>

namespace vpu {

MergeFullyConnectedContentsByChannels::MergeFullyConnectedContentsByChannels(const std::vector<DataContent::CPtr> contents,
                                                                             const std::vector<DataDesc> inDescs,
                                                                             const DataDesc& resDesc) :
        _contents(contents), _inDescs(inDescs), _resDesc(resDesc) {}

size_t MergeFullyConnectedContentsByChannels::byteSize() const {
    return checked_cast<size_t>(_resDesc.totalDimSize()) *
           checked_cast<size_t>(_resDesc.elemSize());
}

void MergeFullyConnectedContentsByChannels::fillTempBuf(void* temp) const {
    IE_ASSERT(!_contents.empty());
    // vpu::DataNode has content and vpu::DataDesc with dimensions' vector
    // content has dimensions's vector as well
    // they can be different so we extract channels number from contents
    const auto dstC = std::accumulate(_inDescs.begin(), _inDescs.end(), 0, [](int reduction, const DataDesc& desc) {
        return reduction + desc.dims()[Dim::C];});

    for (std::size_t i = 0, dstChannelsOffset = 0; i < _inDescs.size(); ++i) {
        const auto& content = _contents[i];
        const auto& srcDesc = _inDescs[i];

        const auto& srcDims = srcDesc.dims();
        const auto& elemSize = srcDesc.elemSize();

        const auto N = srcDims.get(Dim::N, 1);
        const auto H = srcDims.get(Dim::H, 1);
        const auto W = srcDims.get(Dim::W, 1) * elemSize;

        const auto& srcC = srcDims[Dim::C];

        const auto src = content->get<uint8_t>();
        auto dst = static_cast<uint8_t*>(temp);

        InferenceEngine::parallel_for4d(N, srcC, H, W, [dstChannelsOffset, N, H, W, src, dst, srcC, dstC](int n, int c, int h, int w) {
            const auto& srcc = c;
            const auto& dstc = dstChannelsOffset + c;

            const auto& srcOffset = n * H * W * srcC + srcc * H * W + h * W + w;
            const auto& dstOffset = n * H * W * dstC + dstc * H * W + h * W + w;
            dst[dstOffset] = src[srcOffset];
        });

        dstChannelsOffset += srcC;
    }
}

} // namespace vpu
