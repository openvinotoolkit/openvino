// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <utility>
#include <vector>

#include "ie_blob.h"

namespace InferenceEngine {
Blob::Ptr Blob::createROI(const ROI& roi) const {
    if (getTensorDesc().getLayout() == Layout::NCHW || getTensorDesc().getLayout() == Layout::NHWC) {
        return createROI({roi.id, 0, roi.posY, roi.posX},
                         {roi.id + 1, getTensorDesc().getDims()[1], roi.posY + roi.sizeY, roi.posX + roi.sizeX});
    }
    IE_THROW(NotImplemented) << "createROI is not implemented for current type of Blob";
}

Blob::Ptr Blob::createROI(const std::vector<std::size_t>& begin, const std::vector<std::size_t>& end) const {
    IE_THROW(NotImplemented) << "createROI is not implemented for current type of Blob or roi";
}

Blob::Ptr make_shared_blob(const Blob::Ptr& inputBlob, const ROI& roi) {
    return inputBlob->createROI(roi);
}

Blob::Ptr make_shared_blob(const Blob::Ptr& inputBlob,
                           const std::vector<std::size_t>& begin,
                           const std::vector<std::size_t>& end) {
    return inputBlob->createROI(begin, end);
}

//
// RTTI
//

Blob::~Blob() {}
MemoryBlob::~MemoryBlob() {}

template <typename T, typename U>
TBlob<T, U>::~TBlob() {
    free();
}

template class INFERENCE_ENGINE_API_CLASS(TBlob<float>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<double>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<int8_t>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<uint8_t>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<int16_t>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<uint16_t>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<int32_t>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<uint32_t>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<long>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<long long>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<unsigned long>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<unsigned long long>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<bool>);
template class INFERENCE_ENGINE_API_CLASS(TBlob<char>);

}  // namespace InferenceEngine
