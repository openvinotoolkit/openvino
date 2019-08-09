// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <ie_compound_blob.h>
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

class CompoundBlobTests : public ::testing::Test {
protected:
    Blob::Ptr _test_blob;
    using BlobPtrs = std::vector<Blob::Ptr>;
    using MemoryBlobPtrs = std::vector<MemoryBlob::Ptr>;

public:
    void verifyCompoundBlob(const Blob::Ptr& blob) {
        // verify basic assumptions about a compound blob
        ASSERT_NE(nullptr, blob);
        ASSERT_TRUE(blob->is<CompoundBlob>());
        CompoundBlob::Ptr compound_blob = as<CompoundBlob>(blob);
        ASSERT_NE(nullptr, compound_blob);
        EXPECT_EQ(compound_blob.get(), blob->as<CompoundBlob>());  // shared object == raw ptr
        EXPECT_EQ(0, compound_blob->element_size());
        EXPECT_EQ(nullptr, compound_blob->buffer());
        EXPECT_EQ(nullptr, compound_blob->cbuffer());
        EXPECT_GT(compound_blob->size(), 0);
        EXPECT_NE(nullptr, compound_blob->getBlob(0));
    }

    void verifyCompoundBlob(Blob::Ptr blob, const BlobPtrs& underlying_blobs) {
        verifyCompoundBlob(blob);

        // check that the compound blob contains a vector of provided underlying blobs
        CompoundBlob::Ptr compound_blob = as<CompoundBlob>(blob);
        EXPECT_EQ(compound_blob.get(), blob->as<CompoundBlob>());  // shared object == raw ptr
        ASSERT_EQ(underlying_blobs.size(), compound_blob->size());
        for (size_t i = 0; i < underlying_blobs.size(); ++i) {
            EXPECT_EQ(underlying_blobs[i], compound_blob->getBlob(i));
        }
    }
};

class NV12BlobTests : public CompoundBlobTests {};

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

TEST(BlobConversionTests, canWorkWithMemoryBlob) {
    Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 3, 4, 4}, NCHW));
    ASSERT_TRUE(blob->is<MemoryBlob>());
    ASSERT_FALSE(blob->is<CompoundBlob>());
    ASSERT_NE(nullptr, as<MemoryBlob>(blob));
    ASSERT_EQ(nullptr, as<CompoundBlob>(blob));
    ASSERT_EQ(as<MemoryBlob>(blob).get(), blob->as<MemoryBlob>());
    ASSERT_EQ(as<CompoundBlob>(blob).get(), blob->as<CompoundBlob>());
}

TEST(BlobConversionTests, canWorkWithConstMemoryBlob) {
    Blob::CPtr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 3, 4, 4}, NCHW));
    ASSERT_TRUE(blob->is<MemoryBlob>());
    ASSERT_FALSE(blob->is<CompoundBlob>());
    ASSERT_NE(nullptr, as<MemoryBlob>(blob));
    ASSERT_EQ(nullptr, as<CompoundBlob>(blob));
    ASSERT_EQ(as<MemoryBlob>(blob).get(), blob->as<MemoryBlob>());
    ASSERT_EQ(as<CompoundBlob>(blob).get(), blob->as<CompoundBlob>());
}

TEST(BlobConversionTests, canWorkWithTBlob) {
    Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 3, 4, 4}, NCHW));
    ASSERT_TRUE(blob->is<TBlob<uint8_t>>());
    ASSERT_FALSE(blob->is<TBlob<float>>());
    ASSERT_FALSE(blob->is<CompoundBlob>());
    ASSERT_NE(nullptr, as<TBlob<uint8_t>>(blob));
    ASSERT_EQ(nullptr, as<TBlob<float>>(blob));
    ASSERT_EQ(nullptr, as<CompoundBlob>(blob));
    ASSERT_EQ(as<TBlob<uint8_t>>(blob).get(), blob->as<TBlob<uint8_t>>());
    ASSERT_EQ(as<TBlob<float>>(blob).get(), blob->as<TBlob<float>>());
    ASSERT_EQ(as<CompoundBlob>(blob).get(), blob->as<CompoundBlob>());
}

TEST(BlobConversionTests, canWorkWithConstTBlob) {
    Blob::CPtr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 3, 4, 4}, NCHW));
    ASSERT_TRUE(blob->is<TBlob<uint8_t>>());
    ASSERT_FALSE(blob->is<TBlob<float>>());
    ASSERT_FALSE(blob->is<CompoundBlob>());
    ASSERT_NE(nullptr, as<TBlob<uint8_t>>(blob));
    ASSERT_EQ(nullptr, as<TBlob<float>>(blob));
    ASSERT_EQ(nullptr, as<CompoundBlob>(blob));
    ASSERT_EQ(as<TBlob<uint8_t>>(blob).get(), blob->as<TBlob<uint8_t>>());
    ASSERT_EQ(as<TBlob<float>>(blob).get(), blob->as<TBlob<float>>());
    ASSERT_EQ(as<CompoundBlob>(blob).get(), blob->as<CompoundBlob>());
}

TEST(BlobConversionTests, canWorkWithCompoundBlob) {
    Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 3, 4, 4}, NCHW));
    Blob::Ptr cblob = make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>({blob}));
    ASSERT_TRUE(cblob->is<CompoundBlob>());
    ASSERT_FALSE(cblob->is<MemoryBlob>());
    ASSERT_NE(nullptr, as<CompoundBlob>(cblob));
    ASSERT_EQ(nullptr, as<MemoryBlob>(cblob));
    ASSERT_EQ(as<CompoundBlob>(cblob).get(), cblob->as<CompoundBlob>());
    ASSERT_EQ(as<MemoryBlob>(cblob).get(), cblob->as<MemoryBlob>());
}

TEST(BlobConversionTests, canWorkWithConstCompoundBlob) {
    Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 3, 4, 4}, NCHW));
    Blob::CPtr cblob = make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>({blob}));
    ASSERT_TRUE(cblob->is<CompoundBlob>());
    ASSERT_FALSE(cblob->is<MemoryBlob>());
    ASSERT_NE(nullptr, as<CompoundBlob>(cblob));
    ASSERT_EQ(nullptr, as<MemoryBlob>(cblob));
    ASSERT_EQ(as<CompoundBlob>(cblob).get(), cblob->as<CompoundBlob>());
    ASSERT_EQ(as<MemoryBlob>(cblob).get(), cblob->as<MemoryBlob>());
}

TEST(BlobConversionTests, blobSharesOwnershipOnCast) {
    static constexpr const uint8_t stored_value = 123;
    TBlob<uint8_t>::Ptr tblob;
    {
        Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1}, HW));
        ASSERT_EQ(1, blob.use_count());
        ASSERT_TRUE(blob->is<TBlob<uint8_t>>());
        tblob = as<TBlob<uint8_t>>(blob);
        ASSERT_NE(nullptr, tblob);
        ASSERT_EQ(2, blob.use_count());
        ASSERT_EQ(2, tblob.use_count());
        tblob->allocate();
        tblob->data()[0] = stored_value;
        ASSERT_EQ(stored_value, tblob->data()[0]);
    }
    ASSERT_EQ(1, tblob.use_count());
    ASSERT_NE(nullptr, tblob);
    ASSERT_EQ(stored_value, tblob->data()[0]);
}

TEST_F(BlobTests, canCreateBlobUsingDefaultAllocator)
{
    SizeVector v = {1,2,3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), free(_)).Times(1);

    {
        TBlob<float> blob({ Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
        blob.allocate();
    }
}

TEST_F(BlobTests, secondAllocateWontMemLeak) {
    SizeVector v = {1,2,3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).Times(2).WillRepeatedly(Return((void*)1));
    EXPECT_CALL(*allocator.get(), free(_)).Times(2).WillRepeatedly(Return(true));

    {
        TBlob<float> blob({ Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
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

    TBlob<float> blob({ Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
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

    TBlob<float> blob({ Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
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

    TBlob<float> blob({ Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
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

    TBlob<float> blob({ Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
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

    TBlob<uint8_t > blob({ Precision::U8, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
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

    TBlob<uint8_t > blob({ Precision::U8, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));

    TBlob<uint8_t > newBlob(std::move(blob));

    ASSERT_EQ(newBlob.size(), 1 * 2 * 3);
    ASSERT_EQ(newBlob.getTensorDesc().getDims()[0], 1);
    ASSERT_EQ(newBlob.getTensorDesc().getDims()[1], 2);
    ASSERT_EQ(newBlob.getTensorDesc().getDims()[2], 3);
}

// test will be removed in R3
IE_SUPPRESS_DEPRECATED_START
TEST_F(BlobTests, canSetAfterFree)
{
    SizeVector v = {1, 3};
    TBlob<uint8_t> blob({ Precision::U8, v, HW });
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
    TBlob<uint8_t> blob({ Precision::U8, v, HW });
    blob.deallocate();
    ASSERT_NO_THROW(blob.set({1,2,3}));
}
IE_SUPPRESS_DEPRECATED_END


TEST_F(BlobTests, canCopyBlob)
{
    SizeVector v = {1, 3};
    TBlob<uint8_t> blob({ Precision::U8, v, HW });
    blob.allocate();
    blob.data()[0] = 1;
    blob.data()[1] = 2;
    blob.data()[2] = 3;

    TBlob<uint8_t> blob2(blob);

    ASSERT_EQ(blob2.getTensorDesc().getDims().size(),  blob.getTensorDesc().getDims().size());
    ASSERT_EQ(blob2.getTensorDesc().getDims()[0],  blob.getTensorDesc().getDims()[0]);
    ASSERT_EQ(blob2.getTensorDesc().getDims()[1],  blob.getTensorDesc().getDims()[1]);
    ASSERT_EQ(blob2.size(),  blob.size());
    ASSERT_EQ(blob2.data()[0],  blob.data()[0]);
    ASSERT_EQ(blob2.data()[1],  blob.data()[1]);
    ASSERT_EQ(blob2.data()[2],  blob.data()[2]);
}

TEST_F(BlobTests, canCompareToNullPtrWithoutDereferencing) {
    SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    TBlob<uint8_t> blob({ Precision::U8, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));

    ASSERT_TRUE(blob.readOnly() == nullptr);
    ASSERT_TRUE(blob.data() == nullptr);
    ASSERT_TRUE(blob.buffer() == nullptr);

    ASSERT_TRUE(nullptr == blob.readOnly());
    ASSERT_TRUE(nullptr == blob.data());
    ASSERT_TRUE(nullptr == blob.buffer());
}

TEST_F(BlobTests, canCreateBlob) {
    InferenceEngine::SizeVector size = { 1, 1, 1 };
    InferenceEngine::TBlob<float> blob({ Precision::FP32, size, CHW });
    ASSERT_NE(blob.size(), 0);
    ASSERT_EQ(blob.buffer(), nullptr);
}

TEST_F(BlobTests, canAllocateBlob) {
    InferenceEngine::SizeVector size = { 1, 1, 1 };
    InferenceEngine::TBlob<float> blob({ Precision::FP32, size, CHW });
    blob.allocate();
    float* buffer = static_cast<float*>(blob.data());
    ASSERT_NE(buffer, nullptr);
}

TEST_F(BlobTests, canDeallocateBlob) {
    InferenceEngine::SizeVector size = { 1, 1, 1 };
    InferenceEngine::TBlob<float> blob({ Precision::FP32, size, CHW });
    blob.allocate();
    blob.deallocate();
    ASSERT_EQ(nullptr, blob.data().as<float*>());
}

TEST_F(BlobTests, canCreateBlobWithoutDims) {
    InferenceEngine::TBlob<float> blob(TensorDesc(Precision::FP32, NCHW));
    ASSERT_EQ(blob.getTensorDesc().getDims().size(), 0);
}

// test will be removed in R3
IE_SUPPRESS_DEPRECATED_START
TEST_F(BlobTests, canSetToBlobWithoutDims) {
    InferenceEngine::TBlob<float> blob(TensorDesc(Precision::FP32, C));
    std::vector<float> data = { 1.0f, 2.0f, 3.0f };
    blob.set(data);
    ASSERT_EQ(blob.byteSize(), data.size() * sizeof(float));
}
IE_SUPPRESS_DEPRECATED_END

TEST_F(BlobTests, canReadDataFromConstBlob) {
    InferenceEngine::TBlob<float> blob({ Precision::FP32, { 1, 1, 1 }, CHW });
    blob.allocate();
    blob.data()[0] = 1.0f;
    InferenceEngine::TBlob<float> const blob2 = blob;
    const float* buf = blob2.readOnly();
    ASSERT_NE(buf, nullptr);
}

TEST_F(BlobTests, canMakeSharedBlob) {
    InferenceEngine::SizeVector size = { 1, 1, 1 };
    InferenceEngine::TBlob<float>::Ptr blob1 = InferenceEngine::make_shared_blob<float>(TensorDesc(Precision::FP32, NCHW));
    InferenceEngine::TBlob<float>::Ptr blob2 = InferenceEngine::make_shared_blob<float>({ Precision::FP32, size, CHW });
    InferenceEngine::TBlob<float>::Ptr blob3
        = InferenceEngine::make_shared_blob<float>({ Precision::FP32, { 0 }, C });
    ASSERT_EQ(blob1->size(), 0);
    ASSERT_EQ(blob2->size(), 1);
    ASSERT_EQ(blob3->size(), 0);
}

TEST_F(BlobTests, cannotCreateBlobWithIncorrectPrecision) {
    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP16, {1, 3, 227, 227}, Layout::NCHW);
    ASSERT_THROW(InferenceEngine::make_shared_blob<float>(desc), InferenceEngine::details::InferenceEngineException);
}

TEST_F(BlobTests, canUseBlobInMoveSemantics) {

    TBlob<float> b(TensorDesc(Precision::FP32, C));

    b.getTensorDesc().setDims({3});
    b.allocate();
    b.data()[0] = 1.0f;
    b.data()[1] = 2.0f;
    b.data()[2] = 3.0f;

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
    std::vector<float> v({1.0f, 2.0f, 3.0f});
    for (auto e: *make_shared_blob<float>(TensorDesc(Precision::FP32, C), &v[0], v.size())) {
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
        auto  b = make_shared_blob<float>(TensorDesc(Precision::FP32, {1, 2}, HW), input);
        auto i = b->begin();
        ASSERT_NEAR(*i, 0.1, 0.00001);
        i++;
        ASSERT_NEAR(*i, 0.2, 0.00001);
        i++;
        ASSERT_EQ(i, b->end());

        ASSERT_EQ(&*b->begin(), input);
    }
}

TEST_F(BlobTests, preAllocatorWillnotWorkIfPtrNotAlocated) {
   ASSERT_ANY_THROW(TBlob<float>({ Precision::FP32, {1}, C }, nullptr));
}

// test will be removed in R3
IE_SUPPRESS_DEPRECATED_START
TEST_F(BlobTests, cannotIncreaseSizeOfPreallocated) {

    float input[] = {0.1f, 0.2f, 0.3f};
    auto  b = make_shared_blob({ Precision::FP32, {1, 2}, HW }, input);

    b->Resize({1,3});
    //since allocator isno't releasing, user have to be carefull that this still use old array
    ASSERT_EQ(nullptr, b->buffer().as<float*>());

    b->Resize({1,1});
    ASSERT_NE(nullptr, b->buffer().as<float*>());

    b->Resize({1,2});
    ASSERT_NE(nullptr, b->buffer().as<float*>());
}


TEST_F(BlobTests, canAcceptpreallocatedSize) {

    float input[] = {0.1f, 0.2f, 0.3f};
    auto  b = make_shared_blob({ Precision::FP32, {1, 2}, HW }, input, 100);
    b->Resize({1,101});
    //since allocator isn't releasing, user have to be carefull that this still use old array
    ASSERT_EQ(nullptr, b->buffer().as<float*>());

    b->Resize({1,100});
    ASSERT_NE(nullptr, b->buffer().as<float*>());
}
IE_SUPPRESS_DEPRECATED_END

TEST_F(BlobTests, ifBlobCannotReleaseItWillReuseOldMemory) {

    SizeVector v = {1,2,3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float))).WillOnce(Return((void*)1));
    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 4 * sizeof(float))).WillOnce(Return((void*)1));
    EXPECT_CALL(*allocator.get(), free(_)).WillRepeatedly(Return(false));

    {
        // this part of test will be removed in R3
        IE_SUPPRESS_DEPRECATED_START
        TBlob<float> blob({ Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
        blob.allocate();
        blob.Resize({1,2,4});
        IE_SUPPRESS_DEPRECATED_END
    }
}

TEST_F(BlobTests, ifBlobCannotReleaseItWillReuseOldMemoryOnlyIfAllocated) {

    SizeVector v = {1,2,3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), free(_)).WillRepeatedly(Return(false));

    {
        // this part of test will be removed in R3
        IE_SUPPRESS_DEPRECATED_START
        TBlob<float> blob({ Precision::FP32, v, CHW }, dynamic_pointer_cast<IAllocator>(allocator));
        blob.Resize({1,2,4});
        IE_SUPPRESS_DEPRECATED_END
    }
}

TEST_F(BlobTests, canModifyDataInRangedFor) {

    SizeVector v = {1,2,3};
    TBlob<int> blob({ Precision::I32, v, CHW });
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

TEST_F(CompoundBlobTests, cannotCreateCompoundBlobFromNullptr) {
    Blob::Ptr valid = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 3, 4, 4}, NCHW));
    EXPECT_THROW(make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>({valid, nullptr})),
        InferenceEngine::details::InferenceEngineException);
}

TEST_F(CompoundBlobTests, canCreateEmptyCompoundBlob) {
    _test_blob = make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>());

    ASSERT_NE(nullptr, _test_blob);
    EXPECT_EQ(0, _test_blob->element_size());
    EXPECT_EQ(nullptr, _test_blob->buffer());
    EXPECT_EQ(nullptr, _test_blob->cbuffer());
    ASSERT_TRUE(_test_blob->is<CompoundBlob>());
    CompoundBlob::Ptr compound_blob = as<CompoundBlob>(_test_blob);
    ASSERT_NE(nullptr, compound_blob);
    EXPECT_EQ(0, compound_blob->size());
    EXPECT_EQ(nullptr, compound_blob->getBlob(0));
}

TEST_F(CompoundBlobTests, canCreateCompoundBlob) {
    // Create a blob with NCHW layout and pass it to compound for test
    Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 3, 4, 4}, NCHW));
    BlobPtrs blobs = {blob};

    _test_blob = make_shared_blob<CompoundBlob>(blobs);
    verifyCompoundBlob(_test_blob, blobs);
}

TEST_F(CompoundBlobTests, cannotCreateCompoundBlobFromCompoundBlob) {
    // Create a blob with NCHW layout and pass it to compound for test. The created compound blob
    // cannot be used to construct another compound blob. Recursive behavior is rejected
    Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 3, 4, 4}, NCHW));

    _test_blob = make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>({blob}));
    verifyCompoundBlob(_test_blob);

    EXPECT_THROW(make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>({blob, _test_blob})),
        InferenceEngine::details::InferenceEngineException);
}

TEST_F(CompoundBlobTests, compoundBlobHoldsCorrectDataInCorrectOrder) {
    // Create a vector of blobs with HW layout and pass it to a compound blob to test if the vector
    // is stored correctly
    static constexpr const uint8_t MAGIC_NUMBER = 23;
    BlobPtrs blobs(5);
    for (size_t i = 0; i < blobs.size(); ++i) {
        blobs[i] = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1}, HW));
        blobs[i]->allocate();
        blobs[i]->buffer().as<uint8_t*>()[0] = static_cast<uint8_t>(i + MAGIC_NUMBER);
    }

    _test_blob = make_shared_blob<CompoundBlob>(blobs);

    verifyCompoundBlob(_test_blob, blobs);

    CompoundBlob::Ptr compound_blob = as<CompoundBlob>(_test_blob);
    EXPECT_EQ(blobs.size(), compound_blob->size());
    for (size_t i = 0; i < compound_blob->size(); ++i) {
        auto blob = compound_blob->getBlob(i);
        ASSERT_NE(nullptr, blob);
        EXPECT_EQ(static_cast<uint8_t>(i + MAGIC_NUMBER), blob->buffer().as<uint8_t*>()[0]);
    }
}

TEST_F(CompoundBlobTests, compoundBlobHoldsReferencesToBlobs) {
    // Create a blob with HW layout and pass it to a compound blob to check that the compound blob
    // holds references to the blob and not a copy of it
    Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1}, HW));
    blob->allocate();
    blob->buffer().as<uint8_t*>()[0] = 12;
    _test_blob = make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>({blob}));

    verifyCompoundBlob(_test_blob);

    CompoundBlob::Ptr compound_blob = as<CompoundBlob>(_test_blob);
    EXPECT_EQ(12, compound_blob->getBlob(0)->buffer().as<uint8_t*>()[0]);
    blob->buffer().as<uint8_t*>()[0] = 34;
    EXPECT_EQ(34, compound_blob->getBlob(0)->buffer().as<uint8_t*>()[0]);
}

TEST_F(CompoundBlobTests, compoundBlobHoldsValidDataWhenUnderlyingBlobIsDestroyed) {
    // Create a scoped blob with HW layout, pass it to compound, and destroy the original scoped
    // blob. Check that the compound blob, which holds a reference to the destroyed blob, still has
    // a valid object
    static constexpr const uint8_t stored_value = 123;
    {
        Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1}, HW));
        blob->allocate();
        blob->buffer().as<uint8_t*>()[0] = stored_value;
        _test_blob = make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>({blob}));
    }

    verifyCompoundBlob(_test_blob);
    CompoundBlob::Ptr compound_blob = as<CompoundBlob>(_test_blob);
    ASSERT_NE(nullptr, compound_blob->getBlob(0));
    EXPECT_EQ(stored_value, compound_blob->getBlob(0)->buffer().as<uint8_t*>()[0]);
}

TEST_F(NV12BlobTests, cannotCreateNV12BlobFromNullptrBlobs) {
    Blob::Ptr valid = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 4, 4}, NHWC));
    EXPECT_THROW(make_shared_blob<NV12Blob>(valid, nullptr),
        InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<NV12Blob>(nullptr, valid),
        InferenceEngine::details::InferenceEngineException);
}

TEST_F(NV12BlobTests, cannotCreateNV12BlobFromCompoundBlobs) {
    Blob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 4, 4}, NHWC));
    auto cblob = make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>({blob}));
    EXPECT_THROW(make_shared_blob<NV12Blob>(cblob, blob),
        InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<NV12Blob>(blob, cblob),
        InferenceEngine::details::InferenceEngineException);
}

TEST_F(NV12BlobTests, cannotCreateNV12BlobFromPlanesWithDifferentElementSize) {
    Blob::Ptr blob_u8 = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 4, 4}, NHWC));
    Blob::Ptr blob_float = make_shared_blob<float>(TensorDesc(Precision::FP32, {1, 2, 2, 2}, NHWC));
    EXPECT_THROW(make_shared_blob<NV12Blob>(blob_u8, blob_float),
        InferenceEngine::details::InferenceEngineException);
}

TEST_F(NV12BlobTests, cannotCreateNV12BlobFromPlanesWithNonU8Precision) {
    Blob::Ptr float_y_blob = make_shared_blob<float>(TensorDesc(Precision::FP32, {1, 1, 4, 4}, NHWC));
    Blob::Ptr float_uv_blob = make_shared_blob<float>(TensorDesc(Precision::FP32, {1, 2, 2, 2}, NHWC));
    EXPECT_THROW(make_shared_blob<NV12Blob>(float_y_blob, float_uv_blob),
        InferenceEngine::details::InferenceEngineException);
}

TEST_F(NV12BlobTests, cannotCreateNV12BlobFromPlanesWithInconsistentBatchSize) {
    Blob::Ptr y = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 4, 4}, NHWC));
    Blob::Ptr uv = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {2, 2, 2, 2}, NHWC));
    EXPECT_THROW(make_shared_blob<NV12Blob>(y, uv), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NV12BlobTests, cannotCreateNV12BlobFromPlanesWithWrongChannelNumber) {
    Blob::Ptr y = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 4, 4}, NHWC));
    Blob::Ptr uv = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 2, 2}, NHWC));
    EXPECT_THROW(make_shared_blob<NV12Blob>(y, y), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<NV12Blob>(uv, uv), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<NV12Blob>(uv, y), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NV12BlobTests, cannotCreateNV12BlobFromPlanesWithWrongWidthRatio) {
    Blob::Ptr y = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 6}, NHWC));
    Blob::Ptr uv0 = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 1, 3}, NHWC));
    Blob::Ptr uv1 = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 5, 3}, NHWC));
    Blob::Ptr uv2 = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 6, 3}, NHWC));
    Blob::Ptr uv3 = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 8, 3}, NHWC));
    EXPECT_THROW(make_shared_blob<NV12Blob>(y, uv0), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<NV12Blob>(y, uv1), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<NV12Blob>(y, uv2), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<NV12Blob>(y, uv3), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NV12BlobTests, cannotCreateNV12BlobFromPlanesWithWrongHeightRatio) {
    Blob::Ptr y = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 6}, NHWC));
    Blob::Ptr uv0 = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 3, 1}, NHWC));
    Blob::Ptr uv1 = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 3, 5}, NHWC));
    Blob::Ptr uv2 = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 3, 6}, NHWC));
    Blob::Ptr uv3 = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 3, 8}, NHWC));
    EXPECT_THROW(make_shared_blob<NV12Blob>(y, uv0), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<NV12Blob>(y, uv1), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<NV12Blob>(y, uv2), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<NV12Blob>(y, uv3), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NV12BlobTests, canCreateNV12BlobFromTwoPlanes) {
    Blob::Ptr y_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 8}, NHWC));
    Blob::Ptr uv_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 3, 4}, NHWC));
    NV12Blob::Ptr nv12_blob = make_shared_blob<NV12Blob>(y_blob, uv_blob);
    verifyCompoundBlob(nv12_blob, {y_blob, uv_blob});
    EXPECT_EQ(y_blob, nv12_blob->y());
    EXPECT_EQ(uv_blob, nv12_blob->uv());
}

TEST_F(NV12BlobTests, canCreateNV12BlobFromTwoMovedPlanes) {
    // Blob::Ptr y_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 8}, NHWC));
    // Blob::Ptr uv_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 3, 4}, NHWC));
    NV12Blob::Ptr nv12_blob = make_shared_blob<NV12Blob>(
        make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 8}, NHWC)),
        make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 3, 4}, NHWC)));
    verifyCompoundBlob(nv12_blob);
}
