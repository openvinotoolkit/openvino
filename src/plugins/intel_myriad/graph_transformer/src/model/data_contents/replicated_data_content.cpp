// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/replicated_data_content.hpp>

#include <vpu/utils/profiling.hpp>

#include <ie_parallel.hpp>
#include <precision_utils.h>

namespace vpu {

ReplicatedContent::ReplicatedContent(float val, int count, const DataDesc& desc) :
        _factor{val}, _count(count), _desc(desc) {}

ReplicatedContent::ReplicatedContent(DataContent::Ptr origContent, int count, const DataDesc& desc) :
        _origContent(origContent), _count(count), _desc(desc) {}

size_t ReplicatedContent::byteSize() const {
    if (!_origContent) {
        return checked_cast<size_t>(_count) * sizeof(fp16_t);
    } else {
        IE_ASSERT(_desc.totalDimSize() % _count == 0);

        return checked_cast<size_t>(_desc.totalDimSize()) * sizeof(fp16_t);
    }
}

void ReplicatedContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(ReplicatedContent);

    auto dstPtr = static_cast<fp16_t*>(tempBuf);

    if (!_origContent) {
        std::fill_n(dstPtr, _count, ie::PrecisionUtils::f32tof16(_factor));
    } else {
        IE_ASSERT(_desc.totalDimSize() % _count == 0);

        auto origCount = _desc.totalDimSize() / _count;
        auto origPtr = _origContent->get<fp16_t>();
        IE_ASSERT(origPtr != nullptr);

        ie::parallel_for(_count, [origPtr, origCount, dstPtr](int i) {
            std::copy_n(origPtr, origCount, dstPtr + i * origCount);
        });
    }
}

DataContent::Ptr replicateContent(float val, int count, const DataDesc& desc) {
    return std::make_shared<ReplicatedContent>(val, count, desc);
}

DataContent::Ptr replicateContent(const DataContent::Ptr& origContent, int count, const DataDesc& desc) {
    return std::make_shared<ReplicatedContent>(origContent, count, desc);
}

} // namespace vpu
