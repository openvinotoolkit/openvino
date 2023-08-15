// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_factory.hpp"

#include <memory>

IE_SUPPRESS_DEPRECATED_START

InferenceEngine::Blob::Ptr make_blob_with_precision(const InferenceEngine::TensorDesc& desc) {
    return make_blob_with_precision(desc.getPrecision(), desc);
}

InferenceEngine::Blob::Ptr make_blob_with_precision(const InferenceEngine::TensorDesc& desc, void* ptr) {
    return make_blob_with_precision(desc.getPrecision(), desc, ptr);
}

InferenceEngine::Blob::Ptr make_blob_with_precision(const InferenceEngine::TensorDesc& desc,
                                                    const std::shared_ptr<InferenceEngine::IAllocator>& alloc) {
    return make_blob_with_precision(desc.getPrecision(), desc, alloc);
}

InferenceEngine::Blob::Ptr make_plain_blob(InferenceEngine::Precision prec, const InferenceEngine::SizeVector dims) {
    return make_blob_with_precision({prec, dims, InferenceEngine::TensorDesc::getLayoutByDims(dims)});
}
