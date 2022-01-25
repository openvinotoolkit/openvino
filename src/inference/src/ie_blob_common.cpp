// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <utility>
#include <vector>

#include "ie_blob.h"
#include "system_allocator.hpp"

namespace InferenceEngine {

void Blob::setShape(const SizeVector& dims) {
    // we don't want to allow setShape for:
    // 1. ROI cases
    {
        size_t denseStride = 1;
        const auto& blockedDims = getTensorDesc().getBlockingDesc().getBlockDims();
        const auto& strides = getTensorDesc().getBlockingDesc().getStrides();

        for (size_t i = 1; i <= strides.size(); i++) {
            if (denseStride != strides[strides.size() - i]) {
                IE_THROW() << "Blob::setShape requires dense blob";
            }
            denseStride *= blockedDims[blockedDims.size() - i];
        }
    }

    // 2. Blobs created on top of preallocated memory
    if (std::dynamic_pointer_cast<InferenceEngine::details::PreAllocator>(getAllocator())) {
        IE_THROW() << "Cannot call setShape for Blobs created on top of preallocated memory.";
    }

    if (properProduct(dims) > properProduct(getTensorDesc().getDims())) {
        // New blob shape requires more memory than old one -- reallocate
        if (!deallocate()) {
            IE_THROW() << "Cannot deallocate blob while an attempt to enlarge blob area in setShape.";
        }

        // Old and new ranks should match as well as layouts
        getTensorDesc().setDims(dims);

        allocate();
        // no way to detect if allocation is successful other than map/unmap
        // that we wouldn't like to do here; but for cases when we use SystemMemoryAllocator
        // we can do it
        if (std::dynamic_pointer_cast<InferenceEngine::SystemMemoryAllocator>(getAllocator())) {
            if (buffer() == nullptr) {
                IE_THROW() << "Failed to allocate memory in Blob::setShape";
            }
        }
    } else {
        // Don't shrink area when new size fit the existing area
        getTensorDesc().setDims(dims);
    }
}

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
