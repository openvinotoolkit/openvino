// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file with helper functions to uniformly create Blob objects
 * @file blob_transform.hpp
 */

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "ie_memcpy.h"
#include "ie_blob.h"
#include "ie_data.h"
#include "ie_preprocess.hpp"

/**
 * @private
 */
template <InferenceEngine::Precision::ePrecision precision>
class BlobFactory {
public:
    using BlobType = typename InferenceEngine::PrecisionTrait<precision>::value_type;

    static InferenceEngine::Blob::Ptr make(const InferenceEngine::TensorDesc& desc) {
        return InferenceEngine::make_shared_blob<BlobType>(desc);
    }
    static InferenceEngine::Blob::Ptr make(const InferenceEngine::TensorDesc& desc, void* ptr) {
        return InferenceEngine::make_shared_blob<BlobType>(desc, reinterpret_cast<BlobType*>(ptr));
    }
    static InferenceEngine::Blob::Ptr make(const InferenceEngine::TensorDesc& desc,
                                           const std::shared_ptr<InferenceEngine::IAllocator>& alloc) {
        return InferenceEngine::make_shared_blob<BlobType>(desc, alloc);
    }
};

/**
 * @private
 */
template <InferenceEngine::Precision::ePrecision precision, class... Args>
InferenceEngine::Blob::Ptr make_shared_blob2(Args&&... args) {
    return BlobFactory<precision>::make(std::forward<Args>(args)...);
}

/**
 * @brief      Creates Blob::Ptr with precision.
 * @ingroup    ie_dev_api_memory 
 *
 * @param[in]  desc  The TensorDesc object
 * @return     A Blob::Ptr pointer
 */
INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob::Ptr)
make_blob_with_precision(const InferenceEngine::TensorDesc& desc);

/**
 * @brief      Makes a blob with precision.
 * @ingroup    ie_dev_api_memory 
 *
 * @param[in]  desc  The TensorDesc object
 * @param      ptr   The pointer to a raw memory
 * @return     A Blob::Ptr pointer
 */
INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob::Ptr)
make_blob_with_precision(const InferenceEngine::TensorDesc& desc, void* ptr);

/**
 * @brief      Makes a blob with precision.
 * @ingroup    ie_dev_api_memory 
 *
 * @param[in]  desc   The description
 * @param[in]  alloc  The IAllocator object
 * @return     A Blob::Ptr pointer
 */
INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob::Ptr)
make_blob_with_precision(const InferenceEngine::TensorDesc& desc,
                         const std::shared_ptr<InferenceEngine::IAllocator>& alloc);

/**
 * @brief      Creates a plain Blob::Ptr
 * @ingroup    ie_dev_api_memory 
 *
 * @param[in]  prec  The Precision value
 * @param[in]  dims  The dims
 * @return     A Blob::Ptr pointer
 */
INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob::Ptr)
make_plain_blob(InferenceEngine::Precision prec, const InferenceEngine::SizeVector dims);

/**
 * @brief      Creates Blob::Ptr with precision
 * @ingroup    ie_dev_api_memory 
 *
 * @param[in]  precision  The precision
 * @param      args       The arguments
 * @tparam     Args       Variadic template arguments
 * @return     A Blob::Ptr pointer
 */
template <class... Args>
InferenceEngine::Blob::Ptr make_blob_with_precision(InferenceEngine::Precision precision, Args&&... args) {
    #define USE_FACTORY(precision)                  \
        case InferenceEngine::Precision::precision: \
            return make_shared_blob2<InferenceEngine::Precision::precision>(std::forward<Args>(args)...);

    switch (precision) {
        USE_FACTORY(FP32);
        USE_FACTORY(FP16);
        USE_FACTORY(Q78);
        USE_FACTORY(I16);
        USE_FACTORY(U8);
        USE_FACTORY(I8);
        USE_FACTORY(U16);
        USE_FACTORY(I32);
        USE_FACTORY(I64);
        USE_FACTORY(U64);
        USE_FACTORY(BIN);
        USE_FACTORY(BOOL);
    default:
        THROW_IE_EXCEPTION << "cannot locate blob for precision: " << precision;
    }
    #undef USE_FACTORY
}

/**
 * @brief Create blob with custom precision
 * @ingroup ie_dev_api_memory 
 * @tparam T - type off underlined elements
 * @tparam Args Variadic template type arguments
 * @param args Arguments
 * @return A Blob::Ptr pointer
 */
template <class T, class... Args>
InferenceEngine::Blob::Ptr make_custom_blob(Args&&... args) {
    return InferenceEngine::make_shared_blob<T>(InferenceEngine::Precision::fromType<T>(), std::forward<Args>(args)...);
}

/**
 * @brief Create blob with custom precision
 * @ingroup ie_dev_api_memory 
 * @tparam T A type off underlined elements
 * @param layout A blob layout
 * @param size A blob size
 * @return A Blob::Ptr pointer
 */
template <class T>
InferenceEngine::Blob::Ptr make_custom_blob(InferenceEngine::Layout layout, InferenceEngine::SizeVector size) {
    return InferenceEngine::make_shared_blob<T>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::fromType<T>(), size, layout));
}

/**
 * @brief Creates a TBlob<> object from a Data node
 * @ingroup ie_dev_api_memory 
 * @param data A reference to a smart pointer of the Data node
 * @return Smart pointer to TBlob<> with the relevant C type to the precision of the data node
 */
INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob::Ptr) CreateBlobFromData(const InferenceEngine::DataPtr& data);

/**
 * @brief Copy data from std::vector to Blob
 * @ingroup ie_dev_api_memory 
 * @tparam T type of data in std::vector
 * @param outputBlob An output blob to copy to
 * @param inputVector An input std::vector to copy from 
 */
template <typename T>
void CopyVectorToBlob(const InferenceEngine::Blob::Ptr outputBlob, const std::vector<T>& inputVector) {
    if (outputBlob->size() != inputVector.size()) THROW_IE_EXCEPTION << "Size mismatch between dims and vector";
    if (outputBlob->element_size() != sizeof(T)) THROW_IE_EXCEPTION << "Element size mismatch between blob and vector";
    ie_memcpy(outputBlob->buffer().as<T*>(), outputBlob->byteSize(), &inputVector[0], inputVector.size() * sizeof(T));
}
