// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <gtest/gtest.h>

using namespace InferenceEngine;

using BlobTests = ::testing::Test;

// Testing TBlob(const TensorDesc& tensorDesc, T* ptr, size_t data_size = 0)
TEST(BlobTests, TBlobThrowsIfPtrForPreAllocatorIsNullPtr) {
    ASSERT_THROW(TBlob<float>({ Precision::FP32, {1}, C }, nullptr),
            InferenceEngine::details::InferenceEngineException);
}

// Testing TBlob(const TensorDesc& tensorDesc, const std::shared_ptr<IAllocator>& alloc)
TEST(BlobTests, TBlobThrowsIfAllocatorIsNullPtr) {
    ASSERT_THROW(TBlob<float>({ Precision::FP32, {1}, C }, std::shared_ptr<IAllocator> ()),
        InferenceEngine::details::InferenceEngineException);
}
