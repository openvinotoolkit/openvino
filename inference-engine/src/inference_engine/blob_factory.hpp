// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include "inference_engine.hpp"

template <InferenceEngine::Precision::ePrecision precision>
class BlobFactory {
 public:
    using BlobType = typename InferenceEngine::PrecisionTrait<precision>::value_type;
    static InferenceEngine::Blob::Ptr make(InferenceEngine::Layout l, InferenceEngine::SizeVector dims) {
        return InferenceEngine::make_shared_blob<BlobType>(precision, l, dims);
    }
    static InferenceEngine::Blob::Ptr make(InferenceEngine::Layout l, InferenceEngine::SizeVector dims, void* ptr) {
        return InferenceEngine::make_shared_blob<BlobType>(precision, l, dims, reinterpret_cast<BlobType*>(ptr));
    }
    static InferenceEngine::Blob::Ptr make(const InferenceEngine::TensorDesc& desc) {
        return InferenceEngine::make_shared_blob<BlobType>(desc);
    }
    static InferenceEngine::Blob::Ptr make(const InferenceEngine::TensorDesc& desc, void* ptr) {
        return InferenceEngine::make_shared_blob<BlobType>(desc, reinterpret_cast<BlobType*>(ptr));
    }
};

template <InferenceEngine::Precision::ePrecision precision, class ... Args> InferenceEngine::Blob::Ptr make_shared_blob2(Args && ... args) {
    return BlobFactory<precision>::make(std::forward<Args>(args) ...);
}

// TODO: customize make_shared_blob2
#define USE_FACTORY(precision)\
    case InferenceEngine::Precision::precision  : return make_shared_blob2<InferenceEngine::Precision::precision>(std::forward<Args>(args) ...);

INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob::Ptr) make_blob_with_precision(const InferenceEngine::TensorDesc& desc);
INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob::Ptr) make_blob_with_precision(const InferenceEngine::TensorDesc& desc, void* ptr);
INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob::Ptr) make_plain_blob(InferenceEngine::Precision prec, const InferenceEngine::SizeVector dims);

INFERENCE_ENGINE_API_CPP(InferenceEngine::Layout) plain_layout(InferenceEngine::SizeVector dims);

template <class ... Args>
InferenceEngine::Blob::Ptr make_blob_with_precision(InferenceEngine::Precision precision, Args &&... args) {
    switch (precision) {
        USE_FACTORY(FP32);
        USE_FACTORY(FP16);
        USE_FACTORY(Q78);
        USE_FACTORY(I16);
        USE_FACTORY(U8);
        USE_FACTORY(I8);
        USE_FACTORY(U16);
        USE_FACTORY(I32);
        default:
            THROW_IE_EXCEPTION << "cannot locate blob for precision: " << precision;
    }
}

#undef USE_FACTORY

/**
 * Create blob with custom precision
 * @tparam T - type off underlined elements
 * @tparam Args
 * @param args
 * @return
 */
template <class T, class ... Args>
InferenceEngine::Blob::Ptr make_custom_blob(Args &&... args) {
    return InferenceEngine::make_shared_blob<T>(InferenceEngine::Precision::fromType<T>(), std::forward<Args>(args) ...);
}

/**
 * @brief Creates a TBlob<> object from a Data node
 * @param Data reference to a smart pointer of the Data node
 * @return Smart pointer to TBlob<> with the relevant C type to the precision of the data node
 */
INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob::Ptr) CreateBlobFromData(const InferenceEngine::DataPtr &data);
