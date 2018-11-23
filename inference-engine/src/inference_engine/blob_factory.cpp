// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "blob_factory.hpp"


InferenceEngine::Blob::Ptr make_blob_with_precision(const InferenceEngine::TensorDesc& desc) {
    return make_blob_with_precision(desc.getPrecision(), desc);
}

InferenceEngine::Blob::Ptr make_blob_with_precision(const InferenceEngine::TensorDesc& desc, void *ptr) {
    return make_blob_with_precision(desc.getPrecision(), desc, ptr);
}

InferenceEngine::Blob::Ptr CreateBlobFromData(const InferenceEngine::DataPtr &data) {
    // TODO Here some decision should be made about the layout.
    // For now we just pass the layout and use conversion to NCHW for ANY.
    InferenceEngine::Layout targetLayout = data->getLayout();
    if (data->getLayout() == InferenceEngine::Layout::ANY) {
        targetLayout = InferenceEngine::Layout::NCHW;
    }

    switch (data->getPrecision()) {
        case InferenceEngine::Precision::FP32:
            return std::make_shared<InferenceEngine::TBlob<float>>(data->getPrecision(), targetLayout, data->getDims());
        case InferenceEngine::Precision::Q78:
        case InferenceEngine::Precision::I16:
        case InferenceEngine::Precision::FP16:
            return std::make_shared<InferenceEngine::TBlob<short>>(data->getPrecision(), targetLayout, data->getDims());
        case InferenceEngine::Precision::U8:
            return std::make_shared<InferenceEngine::TBlob<uint8_t>>(data->getPrecision(), targetLayout, data->getDims());
        case InferenceEngine::Precision::I8:
            return std::make_shared<InferenceEngine::TBlob<int8_t>>(data->getPrecision(), targetLayout, data->getDims());
        case InferenceEngine::Precision::I32:
            return std::make_shared<InferenceEngine::TBlob<int32_t>>(data->getPrecision(), targetLayout, data->getDims());
        default:
            THROW_IE_EXCEPTION << "precision is no set";
    }
}
