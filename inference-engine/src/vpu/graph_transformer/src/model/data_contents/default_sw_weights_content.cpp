// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/default_sw_weights_content.hpp>

#include <vpu/utils/profiling.hpp>
#include <vpu/middleend/sw/utility.hpp>

namespace vpu {

DefaultSwWeightsContent::DefaultSwWeightsContent(const DataContent::Ptr& origContent, const DataDesc& desc) :
        _origContent(origContent), _desc(desc) {
}

size_t DefaultSwWeightsContent::byteSize() const {
    return checked_cast<size_t>(_desc.totalDimSize()) *
           checked_cast<size_t>(_desc.elemSize());
}

void DefaultSwWeightsContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(DefaultSwWeightsContent);

    IE_ASSERT(_desc.type() == DataType::FP16);

    kchw_to_hwck(_origContent->get<fp16_t>(), static_cast<fp16_t*>(tempBuf), _desc);
}

} // namespace vpu
