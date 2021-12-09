// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/mean_contents.hpp>

#include <vpu/utils/profiling.hpp>
#include <vpu/middleend/sw/utility.hpp>

#include <precision_utils.h>

namespace vpu {

//
// MeanImageContent
//

MeanImageContent::MeanImageContent(const ie::PreProcessInfo& info, const DataDesc& desc) : _info(info), _desc(desc) {}

size_t MeanImageContent::byteSize() const {
    size_t countElem = checked_cast<size_t>(_desc.dim(Dim::W) * _desc.dim(Dim::H) * _desc.dim(Dim::C));
    if (_desc.dimsOrder() == DimsOrder::NHWC || _desc.dimsOrder() == DimsOrder::HWC) {
        countElem *= 2;
    }

    return countElem * sizeof(fp16_t);
}

void MeanImageContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(MeanImageContent);

    const size_t numOfChannel = _info.getNumberOfChannels();

    const size_t imagePixels = checked_cast<size_t>(_desc.dim(Dim::W) * _desc.dim(Dim::H));
    const size_t countElem = checked_cast<size_t>(_desc.dim(Dim::W) * _desc.dim(Dim::H) * _desc.dim(Dim::C));

    const auto dstPtr = static_cast<fp16_t*>(tempBuf);

    auto dstPtr2 = dstPtr;
    if (_desc.dimsOrder() == DimsOrder::NHWC || _desc.dimsOrder() == DimsOrder::HWC) {
        dstPtr2 += countElem;
    }

    ie::parallel_for(numOfChannel, [=](size_t i) {
        const auto meanDataBlob = _info[i]->meanData;

        ie::PrecisionUtils::f32tof16Arrays(
                dstPtr2 + i * imagePixels,
                meanDataBlob->buffer().as<const float*>(),
                imagePixels,
                -1.0f);
    });

    if (_desc.dimsOrder() == DimsOrder::NHWC || _desc.dimsOrder() == DimsOrder::HWC) {
        kchw_to_hwck(dstPtr2, dstPtr, _desc);
    }
}

//
// MeanValueContent
//

MeanValueContent::MeanValueContent(const ie::PreProcessInfo& info) : _info(info) {}

size_t MeanValueContent::byteSize() const {
    return _info.getNumberOfChannels() * sizeof(fp16_t);
}

void MeanValueContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(MeanValueContent);

    const auto dstPtr = static_cast<fp16_t*>(tempBuf);

    ie::parallel_for(_info.getNumberOfChannels(), [dstPtr, this](size_t i) {
        dstPtr[i] = ie::PrecisionUtils::f32tof16(-_info[i]->meanValue);
    });
}

} // namespace vpu
