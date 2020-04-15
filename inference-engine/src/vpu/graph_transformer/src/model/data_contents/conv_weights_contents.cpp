// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/conv_weights_contents.hpp>

#include <vpu/middleend/sw/utility.hpp>
#include <vpu/utils/profiling.hpp>

namespace vpu {

//
// ConvIm2ColWeightsContent
//

ConvIm2ColWeightsContent::ConvIm2ColWeightsContent(const DataContent::Ptr& origContent, DataDesc desc) :
        _origContent(origContent), _desc(desc) {}

size_t ConvIm2ColWeightsContent::byteSize() const {
    return checked_cast<size_t>(_desc.totalDimSize()) *
           checked_cast<size_t>(_desc.elemSize());
}

void ConvIm2ColWeightsContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(ConvIm2ColWeightsContent);
    kchw_to_khwc(_origContent->get<fp16_t>(), static_cast<fp16_t*>(tempBuf), _desc);
}

//
// Conv3x3WeightsContent
//

Conv3x3WeightsContent::Conv3x3WeightsContent(const DataContent::Ptr& origContent, DataDesc desc) :
        _origContent(origContent), _desc(desc) {
}

size_t Conv3x3WeightsContent::byteSize() const {
    return checked_cast<size_t>(_desc.totalDimSize()) *
           checked_cast<size_t>(_desc.elemSize());
}

void Conv3x3WeightsContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(Conv3x3WeightsContent);
    kchw_to_hwkc(_origContent->get<fp16_t>(), static_cast<fp16_t*>(tempBuf), _desc);
}

//
// ConvCHWWeightsContent
//

ConvCHWWeightsContent::ConvCHWWeightsContent(const DataContent::Ptr& origContent, DataDesc desc) :
        _origContent(origContent), _desc(desc) {
}

size_t ConvCHWWeightsContent::byteSize() const {
    return checked_cast<size_t>(_desc.totalDimSize()) *
           checked_cast<size_t>(_desc.elemSize());
}

void ConvCHWWeightsContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(ConvCHWWeightsContent);
    kchw_to_hwkc(_origContent->get<fp16_t>(), static_cast<fp16_t*>(tempBuf), _desc);
}

} // namespace vpu
