// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/batch_norm_contents.hpp>

#include <vpu/utils/profiling.hpp>

#include <ie_parallel.hpp>
#include <precision_utils.h>

namespace vpu {

namespace ie = InferenceEngine;

//
// BatchNormalizationWeightsContent
//

BatchNormalizationWeightsContent::BatchNormalizationWeightsContent(const DataContent::Ptr& origContent,
                                                                   float epsilon) :
        _origContent(origContent), _epsilon(epsilon) {}

size_t BatchNormalizationWeightsContent::byteSize() const {
    return _origContent->byteSize();
}

void BatchNormalizationWeightsContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(BatchNormalizationWeightsContent);

    auto srcPtr = _origContent->get<fp16_t>();
    auto dstPtr = static_cast<fp16_t*>(tempBuf);

    ie::parallel_for(_origContent->byteSize() / sizeof(fp16_t), [this, srcPtr, dstPtr](size_t i) {
        float val = ie::PrecisionUtils::f16tof32(srcPtr[i]) + _epsilon;
        val = 1.0f / std::sqrt(val);
        dstPtr[i] = ie::PrecisionUtils::f32tof16(val);
    });
}

//
// BatchNormalizationBiasesContent
//

BatchNormalizationBiasesContent::BatchNormalizationBiasesContent(const DataContent::Ptr& origContent,
                                                                 const DataContent::Ptr& weightsContent) :
        _origContent(origContent), _weightsContent(weightsContent) {}

size_t BatchNormalizationBiasesContent::byteSize() const {
    return _origContent->byteSize();
}

void BatchNormalizationBiasesContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(BatchNormalizationBiasesContent);

    auto origPtr = _origContent->get<fp16_t>();
    auto weightsPtr = _weightsContent->get<fp16_t>();

    auto dstPtr = static_cast<fp16_t*>(tempBuf);

    ie::parallel_for(_origContent->byteSize() / sizeof(fp16_t), [origPtr, weightsPtr, dstPtr](size_t i) {
        // TODO : need to be extracted from IE layer.
        float beta = 0.0f;

        auto wVal = ie::PrecisionUtils::f16tof32(weightsPtr[i]);
        dstPtr[i] = ie::PrecisionUtils::f32tof16(beta - wVal * ie::PrecisionUtils::f16tof32(origPtr[i]));
    });
}

} // namespace vpu
