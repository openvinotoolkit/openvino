// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <gtest/gtest.h>
#include <random>
#include <chrono>

#include "mock_allocator.hpp"

#include <cpp/ie_cnn_net_reader.h>
#include <gmock/gmock-spec-builders.h>

#ifdef WIN32
#define UNUSED
#else
#define UNUSED  __attribute__((unused))
#endif

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;

class BlobTests: public ::testing::Test {
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

struct ScopedTimer
{
    chrono::high_resolution_clock::time_point t0;
    function<void(int)> cb;

    ScopedTimer(function<void(int)> callback)
    : t0(chrono::high_resolution_clock::now())
    , cb(callback)
    {
    }
    ~ScopedTimer(void)
    {
        auto  t1 = chrono::high_resolution_clock::now();
        auto milli = chrono::duration_cast<chrono::microseconds>(t1-t0).count();

        cb((int)milli);
    }
};

TEST_F(BlobTests, canCreateBlobUsingDefaultAllocator)
{
    SizeVector v = {1,2,3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);

    {
        TBlob<float> blob(Precision::FP32, CHW, v, dynamic_pointer_cast<IAllocator>(allocator));
        blob.allocate();
    }
}

TEST_F(BlobTests, secondAllocateWontMemLeak) {
    SizeVector v = {1,2,3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).Times(2).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), free(_)).Times(2).WillRepeatedly(Return(true));

    {
        TBlob<float> blob(Precision::FP32, CHW, v, dynamic_pointer_cast<IAllocator>(allocator));
        blob.allocate();
        blob.allocate();
    }
}


TEST_F(BlobTests, doesNotUnlockIfLockFailed)
{
    SizeVector v = {1,2,3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), lock((void*)1,LOCK_FOR_WRITE)).Times(1);
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);

    TBlob<float> blob(Precision::FP32, CHW, v, dynamic_pointer_cast<IAllocator>(allocator));
    blob.allocate();
    {
        float UNUSED *ptr = blob.data();
    }
}

TEST_F(BlobTests, canAccessDataUsingAllocator)
{
    SizeVector v = {1,2,3};
    auto allocator = createMockAllocator();

    float data[] = {5.f,6.f,7.f};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), lock((void*)1, LOCK_FOR_WRITE)).WillRepeatedly(Return(data));
    EXPECT_CALL(*allocator.get(), unlock((void*)1)).Times(1);
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);

    TBlob<float> blob(Precision::FP32, CHW, v, dynamic_pointer_cast<IAllocator>(allocator));
    blob.allocate();
    {
        float *ptr = blob.data();
        ASSERT_EQ(ptr[2] , 7);
    }

}


TEST_F(BlobTests, canLockReadOnlyDataForRead)
{
    SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    float data[] = {5,6,7};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), lock(_,LOCK_FOR_READ)).WillRepeatedly(Return(data));
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);
    EXPECT_CALL(*allocator.get(), unlock((void*)1)).Times(1);

    TBlob<float> blob(Precision::FP32, CHW, v, dynamic_pointer_cast<IAllocator>(allocator));
    blob.allocate();

    const float *ptr = blob.readOnly();
    ASSERT_EQ(ptr[2] , 7);
}

TEST_F(BlobTests, canAccessDataUsingBufferBaseMethod)
{
    SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    float data[] = {5,6,7};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), lock(_,LOCK_FOR_WRITE)).WillRepeatedly(Return(data));
    EXPECT_CALL(*allocator.get(), unlock((void*)1)).Times(1);
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);

    TBlob<float> blob(Precision::FP32, CHW, v, dynamic_pointer_cast<IAllocator>(allocator));
    blob.allocate();
    auto buffer = blob.buffer();
    float *ptr = (float * )(void*)buffer;
    ASSERT_EQ(ptr[2] , 7);
}

TEST_F(BlobTests, canMoveFromTBlobWithSameType)
{
    SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    uint8_t data[] = {5,6};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(uint8_t))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), lock(_,LOCK_FOR_WRITE)).WillRepeatedly(Return(data));
    EXPECT_CALL(*allocator.get(), unlock((void*)1)).Times(1);
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);

    TBlob<uint8_t > blob(Precision::U8, CHW, v, dynamic_pointer_cast<IAllocator>(allocator));
    blob.allocate();

    TBlob<uint8_t > newBlob(std::move(blob));

    auto buffer = newBlob.buffer();
    uint8_t *ptr = (uint8_t * )(void*)buffer;
    ASSERT_EQ(ptr[0] , data[0]);
}

TEST_F(BlobTests, saveDimsAndSizeAfterMove)
{
    SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    TBlob<uint8_t > blob(Precision::U8, CHW, v, dynamic_pointer_cast<IAllocator>(allocator));

    TBlob<uint8_t > newBlob(std::move(blob));

    ASSERT_EQ(newBlob.size(), 1 * 2 * 3);
    ASSERT_EQ(newBlob.dims()[0], 1);
    ASSERT_EQ(newBlob.dims()[1], 2);
    ASSERT_EQ(newBlob.dims()[2], 3);
}


TEST_F(BlobTests, canSetAfterFree)
{
    SizeVector v = {1, 3};
    TBlob<uint8_t> blob(Precision::U8, HW, v);
    blob.allocate();
    blob.data()[0] = 1;
    blob.data()[1] = 2;
    blob.data()[2] = 3;

    blob.deallocate();
    ASSERT_NO_THROW(blob.set({1,2,3}));
}

TEST_F(BlobTests, canSetAfterFreeNonAllocated)
{
    SizeVector v = {1, 3};
    TBlob<uint8_t> blob(Precision::U8, HW, v);
    blob.deallocate();
    ASSERT_NO_THROW(blob.set({1,2,3}));
}


TEST_F(BlobTests, canCopyBlob)
{
    SizeVector v = {1, 3};
    TBlob<uint8_t> blob(Precision::U8, HW,v);
    blob.allocate();
    blob.data()[0] = 1;
    blob.data()[1] = 2;
    blob.data()[2] = 3;

    TBlob<uint8_t> blob2(blob);

    ASSERT_EQ(blob2.dims().size(),  blob.dims().size());
    ASSERT_EQ(blob2.dims()[0],  blob.dims()[0]);
    ASSERT_EQ(blob2.dims()[1],  blob.dims()[1]);
    ASSERT_EQ(blob2.size(),  blob.size());
    ASSERT_EQ(blob2.data()[0],  blob.data()[0]);
    ASSERT_EQ(blob2.data()[1],  blob.data()[1]);
    ASSERT_EQ(blob2.data()[2],  blob.data()[2]);
}

TEST_F(BlobTests, canCompareToNullPtrWithoutDereferencing) {
    SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    TBlob<uint8_t> blob(Precision::U8, CHW, v, dynamic_pointer_cast<IAllocator>(allocator));

    ASSERT_TRUE(blob.readOnly() == nullptr);
    ASSERT_TRUE(blob.data() == nullptr);
    ASSERT_TRUE(blob.buffer() == nullptr);

    ASSERT_TRUE(nullptr == blob.readOnly());
    ASSERT_TRUE(nullptr == blob.data());
    ASSERT_TRUE(nullptr == blob.buffer());
}

TEST_F(BlobTests, canCreateBlob) {
    InferenceEngine::SizeVector size = { 1, 1, 1 };
    InferenceEngine::TBlob<float> blob(Precision::FP32, CHW, size);
    ASSERT_NE(blob.size(), 0);
    ASSERT_EQ(blob.buffer(), nullptr);
}

TEST_F(BlobTests, canAllocateBlob) {
    InferenceEngine::SizeVector size = { 1, 1, 1 };
    InferenceEngine::TBlob<float> blob(Precision::FP32, CHW, size);
    blob.allocate();
    float* buffer = static_cast<float*>(blob.data());
    ASSERT_NE(buffer, nullptr);
}

TEST_F(BlobTests, canDeallocateBlob) {
    InferenceEngine::SizeVector size = { 1, 1, 1 };
    InferenceEngine::TBlob<float> blob(Precision::FP32, CHW, size);
    blob.allocate();
    blob.deallocate();
    ASSERT_EQ(nullptr, blob.data().as<float*>());
}

TEST_F(BlobTests, canCreateBlobWithoutDims) {
    InferenceEngine::TBlob<float> blob(Precision::FP32, NCHW);
    ASSERT_EQ(blob.dims().size(), 0);
}

TEST_F(BlobTests, canSetToBlobWithoutDims) {
    InferenceEngine::TBlob<float> blob(Precision::FP32, C);
    std::vector<float> data = { 1.0f, 2.0f, 3.0f };
    blob.set(data);
    ASSERT_EQ(blob.byteSize(), data.size() * sizeof(float));
}

TEST_F(BlobTests, canReadDataFromConstBlob) {
    InferenceEngine::TBlob<float> blob(Precision::FP32, CHW, { 1, 1, 1 });
    blob.set({ 1.0f });
    InferenceEngine::TBlob<float> const blob2 = blob;
    const float* buf = blob2.readOnly();
    ASSERT_NE(buf, nullptr);
}

TEST_F(BlobTests, canMakeSharedBlob) {
    InferenceEngine::SizeVector size = { 1, 1, 1 };
    InferenceEngine::TBlob<float>::Ptr blob1 = InferenceEngine::make_shared_blob<float>(Precision::FP32, NCHW);
    InferenceEngine::TBlob<float>::Ptr blob2 = InferenceEngine::make_shared_blob<float>(Precision::FP32, CHW, size);
    InferenceEngine::TBlob<float>::Ptr blob3
        = InferenceEngine::make_shared_blob<float, InferenceEngine::SizeVector >(Precision::FP32, C, { 0 });
    ASSERT_EQ(blob1->size(), 0);
    ASSERT_EQ(blob2->size(), 1);
    ASSERT_EQ(blob3->size(), 0);
}

TEST_F(BlobTests, canUseBlobInMoveSemantics) {

    TBlob<float> b(Precision::FP32, C);
    b.set({1.0f, 2.0f, 3.0f});

    std::vector<float> dump;

    for (const auto & e: b) {
        dump.push_back(e);
    }

    ASSERT_EQ(dump.size(), 3);

    ASSERT_EQ(dump[0], 1.0f);
    ASSERT_EQ(dump[1], 2.0f);
    ASSERT_EQ(dump[2], 3.0f);

}

TEST_F(BlobTests, DISABLED_canUseLockedMemoryAsRvalueReference) {

    std::vector<float> dump;
    for (auto e: *make_shared_blob(Precision::FP32, C, std::vector<float>({1.0f, 2.0f, 3.0f}))) {
        dump.push_back(e);
    }

    ASSERT_EQ(dump.size(), 3);

    ASSERT_EQ(dump[0], 1.0f);
    ASSERT_EQ(dump[1], 2.0f);
    ASSERT_EQ(dump[2], 3.0f);
}

TEST_F(BlobTests, canCreateBlobOnExistedMemory) {

    float input[] = {0.1f, 0.2f, 0.3f};
    {
        auto  b = make_shared_blob<float>(Precision::FP32, HW, {1, 2}, input);
        auto i = b->begin();
        ASSERT_NEAR(*i, 0.1, 0.00001);
        i++;
        ASSERT_NEAR(*i, 0.2, 0.00001);
        i++;
        ASSERT_EQ(i, b->end());

        ASSERT_EQ(&*b->begin(), input);
    }
}

TEST_F(BlobTests, preAllocatorWillDoesntWorkIfPtrNotAlocated) {
   ASSERT_ANY_THROW(TBlob<float>(Precision::FP32, C, {1}, nullptr));
}

TEST_F(BlobTests, cannotIncreaseSizeOfPreallocated) {

    float input[] = {0.1f, 0.2f, 0.3f};
    auto  b = make_shared_blob(Precision::FP32, HW, {1, 2}, input);
    b->Resize({1,3});
    //since allocator isno't releasing, user have to be carefull that this still use old array
    ASSERT_EQ(nullptr, b->buffer().as<float*>());

    b->Resize({1,1});
    ASSERT_NE(nullptr, b->buffer().as<float*>());

    b->Resize({1,2});
    ASSERT_NE(nullptr, b->buffer().as<float*>());
}


TEST_F(BlobTests, ifBlobCannotReleaseItWillReuseOldMemory) {

    SizeVector v = {1,2,3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).WillOnce(Return((void*)1));
    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 4 * sizeof(float))).WillOnce(Return((void*)1));
    EXPECT_CALL(*allocator.get(), free(_)).WillRepeatedly(Return(false));

    {
        TBlob<float> blob(Precision::FP32, CHW, v, dynamic_pointer_cast<IAllocator>(allocator));
        blob.allocate();
        blob.Resize({1,2,4});
    }
}

TEST_F(BlobTests, ifBlobCannotReleaseItWillReuseOldMemoryOnlyIfAllocated) {

    SizeVector v = {1,2,3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), free(_)).WillRepeatedly(Return(false));

    {
        TBlob<float> blob(Precision::FP32, CHW, v, dynamic_pointer_cast<IAllocator>(allocator));
        blob.Resize({1,2,4});
    }
}

TEST_F(BlobTests, canModifyDataInRangedFor) {

    SizeVector v = {1,2,3};
    TBlob<int> blob(Precision::I32, CHW, v);
    blob.allocate();

    for (auto & data : blob) {
        data = 5;
    }

    for(int i=0;i<v.size();i++) {
        ASSERT_EQ(5, blob.data()[i]) << "Mismatch at" << i;
    }
}

TEST_F(BlobTests, makeRoiBlobNchw) {
    // we create main blob with NCHW layout. We will crop ROI from this blob.
    SizeVector dims = {1, 3, 6, 5};  // RGB picture of size (WxH) = 5x6
    Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, NCHW));
    blob->allocate();

    // create ROI blob based on the already created blob
    ROI roi = {0, 2, 1, 2, 4};  // cropped picture with: id = 0, (x,y) = (2,1), sizeX (W) = 2, sizeY (H) = 4
    Blob::Ptr roiBlob = make_shared_blob(blob, roi);

    // check that BlockingDesc is constructed properly for the ROI blob
    SizeVector refDims = {1, 3, 4, 2};
    SizeVector refOrder = {0, 1, 2, 3};
    size_t refOffset = 7;
    SizeVector refStrides = {90, 30, 5, 1};
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getBlockDims(), refDims);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOrder(), refOrder);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOffsetPadding(), refOffset);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getStrides(), refStrides);
}

TEST_F(BlobTests, makeRoiBlobNhwc) {
    // we create main blob with NHWC layout. We will crop ROI from this blob.
    SizeVector dims = {1, 3, 4, 8};  // RGB picture of size (WxH) = 8x4
    Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, NHWC));
    blob->allocate();

    // create ROI blob based on the already created blob
    ROI roi = {0, 3, 2, 5, 2};  // cropped picture with: id = 0, (x,y) = (3,2), sizeX (W) = 5, sizeY (H) = 2
    Blob::Ptr roiBlob = make_shared_blob(blob, roi);

    // check that BlockingDesc is constructed properly for the ROI blob
    SizeVector refDims = {1, 2, 5, 3};
    SizeVector refOrder = {0, 2, 3, 1};
    size_t refOffset = 57;
    SizeVector refStrides = {96, 24, 3, 1};
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getBlockDims(), refDims);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOrder(), refOrder);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOffsetPadding(), refOffset);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getStrides(), refStrides);
}

TEST_F(BlobTests, makeRoiBlobWrongSize) {
    // we create main blob with NCHW layout. We will crop ROI from this blob.
    SizeVector dims = {1, 3, 4, 4};  // RGB picture of size (WxH) = 4x4
    Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, NCHW));
    blob->allocate();

    // try to create ROI blob with wrong size
    ROI roi = {0, 1, 1, 4, 4};  // cropped picture with: id = 0, (x,y) = (1,1), sizeX (W) = 4, sizeY (H) = 4
    ASSERT_THROW(make_shared_blob(blob, roi), InferenceEngine::details::InferenceEngineException);
}
