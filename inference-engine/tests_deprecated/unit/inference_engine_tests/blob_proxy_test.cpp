// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>

#include <ie_blob.h>
#include <ie_blob_proxy.hpp>

#include "unit_test_utils/mocks/mock_allocator.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

#define MAKE_SHORT(l, h) h * 0x100 + l

class BlobProxyTests: public ::testing::Test {
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
    }

    shared_ptr<MockAllocator> createMockAllocator() {
        return shared_ptr<MockAllocator>(new MockAllocator());
    }


public:

};

TEST_F(BlobProxyTests, convertByteBlobToFloat) {
    const int size = 4;
    float test_array[size] = {2.2f, 3.5f, 1.1f, 0.0f};
    TBlob<uint8_t>::Ptr b(new TBlob<uint8_t>({ Precision::U8, {size * sizeof(float)}, C }));
    b->allocate();
    uint8_t *sPtr = (uint8_t *) test_array;
    uint8_t *dPtr = b->data();
    ASSERT_EQ(b->size(), size * sizeof(float));
    for (size_t i = 0; i < b->size(); i++) {
        dPtr[i] = sPtr[i];
    }
}

TEST_F(BlobProxyTests, shouldNotDeAllocate)
{
    SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    TBlob<float> blob({ Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));

    Blob::Ptr spBlob(&blob, [](Blob*) {
        //don't delete
    });

    TBlobProxy<float> proxy(Precision::FP32, C, spBlob, 2, {2});

    EXPECT_EQ(((Blob&)proxy).deallocate(), false);
}


TEST_F(BlobProxyTests, canAccessProxyBlobUsingBaseMethod)
{
    SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    float data[] = {5,6,7,8,9,10};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), lock(_,LOCK_FOR_WRITE)).WillRepeatedly(Return(data));
    EXPECT_CALL(*allocator.get(), unlock(_)).Times(1);
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);

    TBlob<float> blob({ Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
    blob.allocate();

    Blob::Ptr spBlob(&blob, [](Blob*) {
        //don't delete
    });

    TBlobProxy<float> proxy(Precision::FP32, C, spBlob, 2, {2});

    auto proxyBuffer = proxy.buffer();
    float *ptr = (float*)(void*)proxyBuffer;
    ASSERT_EQ(ptr[2] , 9);
}

TEST_F(BlobProxyTests, canAccessProxyBlobUsingHelpers)
{
    SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    float data[] = {5,6,7, 8, 9, 10};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), lock(_,LOCK_FOR_WRITE)).WillOnce(Return(data));
    EXPECT_CALL(*allocator.get(), lock(_,LOCK_FOR_READ)).WillOnce(Return(data));
    EXPECT_CALL(*allocator.get(), unlock(_)).Times(2);
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);

    TBlob<float> blob({Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
    blob.allocate();

    Blob::Ptr spBlob(&blob, [](Blob*) {
        //don't delete
    });

    TBlobProxy<float> proxy(Precision::FP32, C, spBlob, 2, {2});

    auto proxyData = proxy.data();
    float *ptr = (float * )&proxyData[0];
    ASSERT_EQ(ptr[2] , 9);

    auto readOnly = proxy.readOnly();
    ptr = (float * )&readOnly[0];
    ASSERT_EQ(ptr[2] , 9);
}

TEST_F(BlobProxyTests, canCreateProxyBlobFromDifferentBaseBlobType)
{
    SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    uint8_t data[] = {5, 6, 7, 8, 9, 10};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(uint8_t))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), lock(_,LOCK_FOR_READ)).WillOnce(Return(data));
    EXPECT_CALL(*allocator.get(), unlock(_)).Times(1);
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);

    TBlob<uint8_t > blob({ Precision::U8, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
    blob.allocate();

    Blob::Ptr spBlob (&blob, [](Blob*){
        //don't delete
    });

    TBlobProxy<short> proxy(Precision::I16, C, spBlob, 0, {3});

    auto readOnly = proxy.readOnly();
    const short * ptr = readOnly;
    ASSERT_EQ(ptr[0] , MAKE_SHORT(data[0], data[1]));
    ASSERT_EQ(ptr[1] , MAKE_SHORT(data[2], data[3]));
    ASSERT_EQ(ptr[2] , MAKE_SHORT(data[4], data[5]));
}

TEST_F(BlobProxyTests, canNotCreateBlobWithOffsetOfSizeOutOfOriginal) {
    SizeVector v = {1, 1, 3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), alloc(1 * 1 * 3 * sizeof(float))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);

    TBlob<float> blob({ Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
    blob.allocate();

    Blob::Ptr spBlob (&blob, [](Blob*){
        //don't delete
    });

    EXPECT_THROW(TBlobProxy<float>(Precision::FP32, C, spBlob, 0, {4}), InferenceEngineException);
    EXPECT_THROW(TBlobProxy<float>(Precision::FP32, C, spBlob, 3, {1}), InferenceEngineException);
    EXPECT_THROW(TBlobProxy<float>(Precision::FP32, C, spBlob, 2, {2}), InferenceEngineException);
}


TEST_F(BlobProxyTests, canAccessBothArraysAfterProxying)
{
    SizeVector v = {1, 2, 4};
    auto allocator = createMockAllocator();

    uint8_t data[] = {5, 6, 7, 8, 9, 10, 11, 12};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 4 * sizeof(uint8_t))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), lock(_,LOCK_FOR_READ)).Times(2).WillRepeatedly(Return(data));
    EXPECT_CALL(*allocator.get(), unlock(_)).Times(2);
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);

    TBlob<uint8_t> blob({ Precision::U8, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
    blob.allocate();

    Blob::Ptr spBlob (&blob, [](Blob*){
        //don't delete
    });

    TBlobProxy<short> proxy(Precision::I16, C, spBlob, 2, {3});

    auto readOnly = proxy.readOnly();
    short * ptr = (short * )&readOnly[0];
    ASSERT_EQ(ptr[0] , MAKE_SHORT(data[2], data[3]));
    ASSERT_EQ(ptr[1] , MAKE_SHORT(data[4], data[5]));

    auto origBuffer = blob.readOnly();
    const uint8_t* origPtr = origBuffer;

    ASSERT_EQ(origPtr[0] , 5);
    ASSERT_EQ(origPtr[1] , 6);

}

TEST_F(BlobProxyTests, convertTwoByteBlobToFloat) {
    const int size = 4;
    float test_array[size] = {2.2f, 3.5f, 1.1f, 0.0f};
    TBlob<uint16_t>::Ptr b(new TBlob<uint16_t>(TensorDesc(Precision::U16, {size*sizeof(float) / sizeof(uint16_t)}, C)));
    b->allocate();
    uint16_t *sPtr = (uint16_t *) test_array;
    uint16_t *dPtr = b->data();
    ASSERT_EQ(b->byteSize(), size*sizeof(float));
    ASSERT_EQ(b->size(), size*sizeof(float)/sizeof(uint16_t));
    for(size_t i = 0; i < b->size(); i++) {
        dPtr[i] = sPtr[i];
    }

    TBlobProxy<float>::Ptr proxy(new TBlobProxy<float>(Precision::FP32, C, b, sizeof(float) / sizeof(uint16_t), {size - 1}));
    ASSERT_NEAR(3.5f, proxy->data()[0], 0.0001f);
    ASSERT_NEAR(1.1f, proxy->data()[1], 0.0001f);
    ASSERT_NEAR(0.0f, proxy->data()[2], 0.0001f);
    ASSERT_EQ(size - 1, proxy->size());
    ASSERT_EQ(size*sizeof(float) - sizeof(float), proxy->byteSize());
}

TEST_F(BlobProxyTests, throwsIfSmallProxyObjectSize) {
    TBlob<float>::Ptr b(new TBlob<float>(TensorDesc(Precision::FP32, C)));

    b->getTensorDesc().setDims({3});
    b->allocate();
    b->data()[0] = 1.0f;
    b->data()[1] = 2.0f;
    b->data()[2] = 3.0f;

    try {
        TBlobProxy<uint8_t> proxy(Precision::U8, C, b, 0, { b->byteSize() + 1 });
        FAIL() << "Should have failed by now: proxy size is larger than blob size";
    }
    catch (InferenceEngine::details::InferenceEngineException ex) {};
}

TEST_F(BlobProxyTests, canReturnConstantData) {
    TBlob<float>::Ptr b(new TBlob<float>(TensorDesc(Precision::FP32, C)));

    b->getTensorDesc().setDims({3});
    b->allocate();
    b->data()[0] = 1.0f;
    b->data()[1] = 2.0f;
    b->data()[2] = 3.0f;

    TBlobProxy<uint8_t> const proxy(Precision::U8, C, b, 0, { b->byteSize() });
    ASSERT_NE(proxy.cbuffer().as<const void*>(), nullptr);
}

TEST_F(BlobProxyTests, canIterateOverData) {
    TBlob<uint8_t>::Ptr b(new TBlob<uint8_t >(TensorDesc(Precision::FP32, C)));

    b->getTensorDesc().setDims({3});
    b->allocate();
    b->data()[0] = 1.0f;
    b->data()[1] = 2.0f;
    b->data()[2] = 3.0f;

    TBlobProxy<uint8_t> proxy(Precision::U8, C, b, 1, { 2 });
    vector<uint8_t > u8buffer;
    for (auto & element : proxy) {
        u8buffer.push_back(element);
    }
    ASSERT_EQ(2, u8buffer.size());
    ASSERT_EQ(2, u8buffer[0]);
    ASSERT_EQ(3, u8buffer[1]);
}

TEST_F(BlobProxyTests, canIterateOverReadOnly) {
    TBlob<uint8_t>::Ptr b(new TBlob<uint8_t >(TensorDesc(Precision::FP32, C)));

    b->getTensorDesc().setDims({3});
    b->allocate();
    b->data()[0] = 1.0f;
    b->data()[1] = 2.0f;
    b->data()[2] = 3.0f;

    TBlobProxy<uint8_t> const proxy(Precision::U8, C, b, 1, { 2 });
    vector<uint8_t > u8buffer;
    for (auto  element : proxy) {
        u8buffer.push_back(element);
    }
    ASSERT_EQ(2, u8buffer.size());
    ASSERT_EQ(2, u8buffer[0]);
    ASSERT_EQ(3, u8buffer[1]);
}
