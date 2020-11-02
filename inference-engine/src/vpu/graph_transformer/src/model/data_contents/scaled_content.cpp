// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/scaled_content.hpp>

#include <vpu/utils/profiling.hpp>

#include <ie_parallel.hpp>
#include <precision_utils.h>

namespace vpu {

ScaledContent::ScaledContent(const DataContent::Ptr& origContent, float scale) :
        _origContent(origContent), _factor(scale) {
}

size_t ScaledContent::byteSize() const {
    return _origContent->byteSize();
}

void ScaledContent::fillTempBuf(void *tempBuf) const {
    VPU_PROFILE(ScaledContent);

    const auto totalSize = _origContent->byteSize() / sizeof(fp16_t);

    auto srcPtr = _origContent->get<fp16_t>();
    IE_ASSERT(srcPtr != nullptr);

    auto dstPtr = static_cast<fp16_t*>(tempBuf);

    ie::parallel_for(totalSize, [this, srcPtr, dstPtr](size_t i) {
        dstPtr[i] = ie::PrecisionUtils::f32tof16(ie::PrecisionUtils::f16tof32(srcPtr[i]) * _factor);
    });
}

DataContent::Ptr scaleContent(const DataContent::Ptr& origContent, float scale) {
    return std::make_shared<ScaledContent>(origContent, scale);
}

} // namespace vpu
