// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-spec-builders.h>
#include <gtest/gtest.h>
#include <ie_blob.h>

#include "openvino/runtime/make_tensor.hpp"
#include "unit_test_utils/mocks/mock_allocator.hpp"

IE_SUPPRESS_DEPRECATED_START

class BlobTests : public ::testing::Test {
protected:
    std::shared_ptr<MockAllocator> createMockAllocator() {
        return std::shared_ptr<MockAllocator>(new MockAllocator());
    }
};

// Testing TBlob(const TensorDesc& tensorDesc, T* ptr, size_t data_size = 0)
TEST_F(BlobTests, TBlobThrowsIfPtrForPreAllocatorIsNullPtr) {
    ASSERT_THROW(InferenceEngine::TBlob<float>({InferenceEngine::Precision::FP32, {1}, InferenceEngine::C}, nullptr),
                 InferenceEngine::Exception);
}

// Testing TBlob(const TensorDesc& tensorDesc, const std::std::shared_ptr<IAllocator>& alloc)
TEST_F(BlobTests, TBlobThrowsIfAllocatorIsNullPtr) {
    ASSERT_THROW(InferenceEngine::TBlob<float>({InferenceEngine::Precision::FP32, {1}, InferenceEngine::C},
                                               std::shared_ptr<InferenceEngine::IAllocator>()),
                 InferenceEngine::Exception);
}

TEST_F(BlobTests, canCreateBlobUsingDefaultAllocator) {
    InferenceEngine::SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float)))
        .WillRepeatedly(testing::Return(reinterpret_cast<void*>(1)));
    EXPECT_CALL(*allocator.get(), free(::testing::_)).Times(1);

    {
        InferenceEngine::TBlob<float> blob({InferenceEngine::Precision::FP32, v, InferenceEngine::CHW},
                                           std::dynamic_pointer_cast<InferenceEngine::IAllocator>(allocator));
        blob.allocate();
    }
}

TEST_F(BlobTests, secondAllocateWontMemLeak) {
    InferenceEngine::SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float)))
        .Times(2)
        .WillRepeatedly(testing::Return(reinterpret_cast<void*>(1)));
    EXPECT_CALL(*allocator.get(), free(::testing::_)).Times(2).WillRepeatedly(testing::Return(true));

    {
        InferenceEngine::TBlob<float> blob({InferenceEngine::Precision::FP32, v, InferenceEngine::CHW},
                                           std::dynamic_pointer_cast<InferenceEngine::IAllocator>(allocator));
        blob.allocate();
        blob.allocate();
    }
}

TEST_F(BlobTests, doesNotUnlockIfLockFailed) {
    InferenceEngine::SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float)))
        .WillRepeatedly(testing::Return(reinterpret_cast<void*>(1)));
    EXPECT_CALL(*allocator.get(), lock(reinterpret_cast<void*>(1), InferenceEngine::LOCK_FOR_WRITE)).Times(1);
    EXPECT_CALL(*allocator.get(), free(::testing::_)).Times(1);

    InferenceEngine::TBlob<float> blob({InferenceEngine::Precision::FP32, v, InferenceEngine::CHW},
                                       std::dynamic_pointer_cast<InferenceEngine::IAllocator>(allocator));
    blob.allocate();
    {
        float* ptr = blob.data();
        (void)ptr;
    }
}

TEST_F(BlobTests, canAccessDataUsingAllocator) {
    InferenceEngine::SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    float data[] = {5.f, 6.f, 7.f};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float)))
        .WillRepeatedly(testing::Return(reinterpret_cast<void*>(1)));
    EXPECT_CALL(*allocator.get(), lock(reinterpret_cast<void*>(1), InferenceEngine::LOCK_FOR_WRITE))
        .WillRepeatedly(testing::Return(data));
    EXPECT_CALL(*allocator.get(), unlock(reinterpret_cast<void*>(1))).Times(1);
    EXPECT_CALL(*allocator.get(), free(::testing::_)).Times(1);

    InferenceEngine::TBlob<float> blob({InferenceEngine::Precision::FP32, v, InferenceEngine::CHW},
                                       std::dynamic_pointer_cast<InferenceEngine::IAllocator>(allocator));
    blob.allocate();
    {
        float* ptr = blob.data();
        ASSERT_EQ(ptr[2], 7);
    }
}

TEST_F(BlobTests, canLockReadOnlyDataForRead) {
    InferenceEngine::SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    float data[] = {5, 6, 7};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float)))
        .WillRepeatedly(testing::Return(reinterpret_cast<void*>(1)));
    EXPECT_CALL(*allocator.get(), lock(::testing::_, InferenceEngine::LOCK_FOR_READ))
        .WillRepeatedly(testing::Return(data));
    EXPECT_CALL(*allocator.get(), free(::testing::_)).Times(1);
    EXPECT_CALL(*allocator.get(), unlock(reinterpret_cast<void*>(1))).Times(1);

    InferenceEngine::TBlob<float> blob({InferenceEngine::Precision::FP32, v, InferenceEngine::CHW},
                                       std::dynamic_pointer_cast<InferenceEngine::IAllocator>(allocator));
    blob.allocate();

    const float* ptr = blob.readOnly();
    ASSERT_EQ(ptr[2], 7);
}

TEST_F(BlobTests, canAccessDataUsingBufferBaseMethod) {
    InferenceEngine::SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    float data[] = {5, 6, 7};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(float)))
        .WillRepeatedly(testing::Return(reinterpret_cast<void*>(1)));
    EXPECT_CALL(*allocator.get(), lock(::testing::_, InferenceEngine::LOCK_FOR_WRITE))
        .WillRepeatedly(testing::Return(data));
    EXPECT_CALL(*allocator.get(), unlock(reinterpret_cast<void*>(1))).Times(1);
    EXPECT_CALL(*allocator.get(), free(::testing::_)).Times(1);

    InferenceEngine::TBlob<float> blob({InferenceEngine::Precision::FP32, v, InferenceEngine::CHW},
                                       std::dynamic_pointer_cast<InferenceEngine::IAllocator>(allocator));
    blob.allocate();
    auto buffer = blob.rwmap();
    const float* ptr = buffer.as<const float*>();
    ASSERT_EQ(ptr[2], 7);
}

TEST_F(BlobTests, canMoveFromTBlobWithSameType) {
    InferenceEngine::SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    uint8_t data[] = {5, 6};

    EXPECT_CALL(*allocator.get(), alloc(1 * 2 * 3 * sizeof(uint8_t)))
        .WillRepeatedly(testing::Return(reinterpret_cast<void*>(1)));
    EXPECT_CALL(*allocator.get(), lock(::testing::_, InferenceEngine::LOCK_FOR_WRITE))
        .WillRepeatedly(testing::Return(data));
    EXPECT_CALL(*allocator.get(), unlock(reinterpret_cast<void*>(1))).Times(1);
    EXPECT_CALL(*allocator.get(), free(::testing::_)).Times(1);

    InferenceEngine::TBlob<uint8_t> blob({InferenceEngine::Precision::U8, v, InferenceEngine::CHW},
                                         std::dynamic_pointer_cast<InferenceEngine::IAllocator>(allocator));
    blob.allocate();

    InferenceEngine::TBlob<uint8_t> newBlob(std::move(blob));

    auto buffer = newBlob.rwmap();
    uint8_t* ptr = buffer.as<uint8_t*>();
    ASSERT_EQ(ptr[0], data[0]);
}

TEST_F(BlobTests, saveDimsAndSizeAfterMove) {
    InferenceEngine::SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    InferenceEngine::TBlob<uint8_t> blob({InferenceEngine::Precision::U8, v, InferenceEngine::CHW},
                                         std::dynamic_pointer_cast<InferenceEngine::IAllocator>(allocator));

    InferenceEngine::TBlob<uint8_t> newBlob(std::move(blob));

    ASSERT_EQ(newBlob.size(), 1 * 2 * 3);
    ASSERT_EQ(newBlob.getTensorDesc().getDims()[0], 1);
    ASSERT_EQ(newBlob.getTensorDesc().getDims()[1], 2);
    ASSERT_EQ(newBlob.getTensorDesc().getDims()[2], 3);
}

TEST_F(BlobTests, canCopyBlob) {
    InferenceEngine::SizeVector v = {1, 3};
    InferenceEngine::TBlob<uint8_t> blob({InferenceEngine::Precision::U8, v, InferenceEngine::HW});
    blob.allocate();
    blob.data()[0] = 1;
    blob.data()[1] = 2;
    blob.data()[2] = 3;

    InferenceEngine::TBlob<uint8_t> blob2(blob);

    ASSERT_EQ(blob2.getTensorDesc().getDims().size(), blob.getTensorDesc().getDims().size());
    ASSERT_EQ(blob2.getTensorDesc().getDims()[0], blob.getTensorDesc().getDims()[0]);
    ASSERT_EQ(blob2.getTensorDesc().getDims()[1], blob.getTensorDesc().getDims()[1]);
    ASSERT_EQ(blob2.size(), blob.size());
    ASSERT_EQ(blob2.data()[0], blob.data()[0]);
    ASSERT_EQ(blob2.data()[1], blob.data()[1]);
    ASSERT_EQ(blob2.data()[2], blob.data()[2]);
}

TEST_F(BlobTests, canCompareToNullPtrWithoutDereferencing) {
    InferenceEngine::SizeVector v = {1, 2, 3};
    auto allocator = createMockAllocator();

    InferenceEngine::TBlob<uint8_t> blob({InferenceEngine::Precision::U8, v, InferenceEngine::CHW},
                                         std::dynamic_pointer_cast<InferenceEngine::IAllocator>(allocator));

    ASSERT_TRUE(blob.readOnly() == nullptr);
    ASSERT_TRUE(blob.data() == nullptr);
    ASSERT_TRUE(blob.rwmap() == nullptr);

    ASSERT_TRUE(nullptr == blob.readOnly());
    ASSERT_TRUE(nullptr == blob.data());
    ASSERT_TRUE(nullptr == blob.rwmap());
}

TEST_F(BlobTests, canCreateBlob) {
    InferenceEngine::SizeVector size = {1, 1, 1};
    InferenceEngine::TBlob<float> blob({InferenceEngine::Precision::FP32, size, InferenceEngine::CHW});
    ASSERT_NE(blob.size(), 0);
    ASSERT_EQ(blob.rwmap(), nullptr);
}

TEST_F(BlobTests, canAllocateBlob) {
    InferenceEngine::SizeVector size = {1, 1, 1};
    InferenceEngine::TBlob<float> blob({InferenceEngine::Precision::FP32, size, InferenceEngine::CHW});
    blob.allocate();
    float* buffer = static_cast<float*>(blob.data());
    ASSERT_NE(buffer, nullptr);
}

TEST_F(BlobTests, canDeallocateBlob) {
    InferenceEngine::SizeVector size = {1, 1, 1};
    InferenceEngine::TBlob<float> blob({InferenceEngine::Precision::FP32, size, InferenceEngine::CHW});
    blob.allocate();
    blob.deallocate();
    ASSERT_EQ(nullptr, blob.data().as<float*>());
}

TEST_F(BlobTests, canCreateBlobWithoutDims) {
    InferenceEngine::TBlob<float> blob(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, InferenceEngine::NCHW));
    ASSERT_EQ(blob.getTensorDesc().getDims().size(), 0);
}

TEST_F(BlobTests, canReadDataFromConstBlob) {
    InferenceEngine::TBlob<float> blob({InferenceEngine::Precision::FP32, {1, 1, 1}, InferenceEngine::CHW});
    blob.allocate();
    blob.data()[0] = 1.0f;
    InferenceEngine::TBlob<float> const blob2 = blob;
    const float* buf = blob2.readOnly();
    ASSERT_NE(buf, nullptr);
}

TEST_F(BlobTests, canMakeSharedBlob) {
    InferenceEngine::SizeVector size = {1, 1, 1};
    InferenceEngine::TBlob<float>::Ptr blob1 = InferenceEngine::make_shared_blob<float>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, InferenceEngine::NCHW));
    InferenceEngine::TBlob<float>::Ptr blob2 =
        InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, size, InferenceEngine::CHW});
    InferenceEngine::TBlob<float>::Ptr blob3 =
        InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, {0}, InferenceEngine::C});
    InferenceEngine::TBlob<float>::Ptr blob4 =
        InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, size, InferenceEngine::HWC});
    ASSERT_EQ(blob1->size(), 0);
    ASSERT_EQ(blob2->size(), 1);
    ASSERT_EQ(blob3->size(), 0);
    ASSERT_EQ(blob4->size(), 1);
}

TEST_F(BlobTests, cannotCreateBlobWithIncorrectPrecision) {
    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP16, {1, 3, 227, 227}, InferenceEngine::Layout::NCHW);
    ASSERT_THROW(InferenceEngine::make_shared_blob<float>(desc), InferenceEngine::Exception);
}

TEST_F(BlobTests, canUseBlobInMoveSemantics) {
    InferenceEngine::TBlob<float> b(InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, InferenceEngine::C));

    b.getTensorDesc().setDims({3});
    b.allocate();
    b.data()[0] = 1.0f;
    b.data()[1] = 2.0f;
    b.data()[2] = 3.0f;

    std::vector<float> dump;

    for (const auto& e : b) {
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
    auto blob = InferenceEngine::make_shared_blob<float>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, InferenceEngine::C),
        &v[0],
        v.size());
    for (auto e : *blob) {
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
        auto b = InferenceEngine::make_shared_blob<float>(
            InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {1, 2}, InferenceEngine::HW),
            input);
        auto i = b->begin();
        ASSERT_NEAR(*i, 0.1, 0.00001);
        i++;
        ASSERT_NEAR(*i, 0.2, 0.00001);
        i++;
        ASSERT_EQ(i, b->end());

        ASSERT_EQ(&*b->begin(), input);
    }
}

// SetShape
TEST_F(BlobTests, canSetShape) {
    auto b = InferenceEngine::make_shared_blob<float>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {1, 2, 3}, InferenceEngine::ANY));
    b->allocate();

    ASSERT_NO_THROW(b->setShape({4, 5, 6}));

    auto newDims = b->getTensorDesc().getDims();
    ASSERT_EQ(newDims.size(), 3);
    ASSERT_EQ(newDims[0], 4);
    ASSERT_EQ(newDims[1], 5);
    ASSERT_EQ(newDims[2], 6);
}

TEST_F(BlobTests, canModifyDataInRangedFor) {
    InferenceEngine::SizeVector v = {1, 2, 3};
    InferenceEngine::TBlob<int> blob({InferenceEngine::Precision::I32, v, InferenceEngine::CHW});
    blob.allocate();

    for (auto& data : blob) {
        data = 5;
    }

    for (size_t i = 0; i < v.size(); i++) {
        ASSERT_EQ(5, blob.data()[i]) << "Mismatch at" << i;
    }
}

TEST_F(BlobTests, makeRoiBlobNchw) {
    // we create main blob with NCHW layout. We will crop ROI from this blob.
    InferenceEngine::SizeVector dims = {1, 3, 6, 5};  // RGB picture of size (WxH) = 5x6
    InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<uint8_t>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, dims, InferenceEngine::NCHW));
    blob->allocate();

    // create ROI blob based on the already created blob
    InferenceEngine::ROI roi = {0,
                                2,
                                1,
                                2,
                                4};  // cropped picture with: id = 0, (x,y) = (2,1), sizeX (W) = 2, sizeY (H) = 4
    InferenceEngine::Blob::Ptr roiBlob = make_shared_blob(blob, roi);

    // check that BlockingDesc is constructed properly for the ROI blob
    InferenceEngine::SizeVector refDims = {1, 3, 4, 2};
    InferenceEngine::SizeVector refOrder = {0, 1, 2, 3};
    size_t refOffset = 7;
    InferenceEngine::SizeVector refStrides = {90, 30, 5, 1};
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getBlockDims(), refDims);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOrder(), refOrder);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOffsetPadding(), refOffset);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getStrides(), refStrides);
}

TEST_F(BlobTests, makeRoiBlobNhwc) {
    // we create main blob with NHWC layout. We will crop ROI from this blob.
    InferenceEngine::SizeVector dims = {1, 3, 4, 8};  // RGB picture of size (WxH) = 8x4
    InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<uint8_t>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, dims, InferenceEngine::NHWC));
    blob->allocate();

    // create ROI blob based on the already created blob
    InferenceEngine::ROI roi = {0,
                                3,
                                2,
                                5,
                                2};  // cropped picture with: id = 0, (x,y) = (3,2), sizeX (W) = 5, sizeY (H) = 2
    InferenceEngine::Blob::Ptr roiBlob = make_shared_blob(blob, roi);

    // check that BlockingDesc is constructed properly for the ROI blob
    InferenceEngine::SizeVector refDims = {1, 2, 5, 3};
    InferenceEngine::SizeVector refOrder = {0, 2, 3, 1};
    size_t refOffset = 57;
    InferenceEngine::SizeVector refStrides = {96, 24, 3, 1};
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getBlockDims(), refDims);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOrder(), refOrder);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOffsetPadding(), refOffset);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getStrides(), refStrides);
}

TEST_F(BlobTests, makeRoiBlobWrongSize) {
    // we create main blob with NCHW layout. We will crop ROI from this blob.
    InferenceEngine::SizeVector dims = {1, 3, 4, 4};  // RGB picture of size (WxH) = 4x4
    InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<uint8_t>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, dims, InferenceEngine::NCHW));
    blob->allocate();

    // try to create ROI blob with wrong size
    InferenceEngine::ROI roi = {0,
                                1,
                                1,
                                4,
                                4};  // cropped picture with: id = 0, (x,y) = (1,1), sizeX (W) = 4, sizeY (H) = 4
    ASSERT_THROW(make_shared_blob(blob, roi), InferenceEngine::Exception);
}

TEST_F(BlobTests, readRoiBlob) {
    // Create original Blob

    const auto origDesc =
        InferenceEngine::TensorDesc(InferenceEngine::Precision::I32, {1, 3, 4, 8}, InferenceEngine::NCHW);

    const auto origBlob = InferenceEngine::make_shared_blob<int32_t>(origDesc);
    origBlob->allocate();

    // Fill the original Blob

    {
        auto origMemory = origBlob->wmap();
        const auto origPtr = origMemory.as<int32_t*>();
        ASSERT_NE(nullptr, origPtr);

        for (size_t i = 0; i < origBlob->size(); ++i) {
            origPtr[i] = static_cast<int32_t>(i);
        }
    }

    // Create ROI Blob

    const auto roi = InferenceEngine::ROI(0, 4, 2, 4, 2);

    const auto roiBlob = InferenceEngine::as<InferenceEngine::MemoryBlob>(origBlob->createROI(roi));
    ASSERT_NE(nullptr, roiBlob);

    // Read ROI Blob

    {
        const auto roiOffset = roiBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();

        auto roiMemory = roiBlob->rmap();
        auto roiPtr = roiMemory.as<const int32_t*>();
        ASSERT_NE(nullptr, roiPtr);

        // Blob::rmap returns pointer to the original blob start, we have to add ROI offset manually.
        roiPtr += roiOffset;

        for (size_t i = 0; i < roiBlob->size(); ++i) {
            ASSERT_EQ(roiPtr[i], i + roiOffset);
        }
    }
}

/////////////////////////////////////////

TEST_F(BlobTests, makeRangeRoiBlobNchw) {
    // we create main blob with NCHW layout. We will crop ROI from this blob.
    InferenceEngine::SizeVector dims = {1, 3, 6, 5};  // RGB picture of size (WxH) = 5x6
    InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<uint8_t>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, dims, InferenceEngine::NCHW));
    blob->allocate();

    // create ROI blob based on the already created blob
    InferenceEngine::ROI roi = {0,
                                2,
                                1,
                                2,
                                4};  // cropped picture with: id = 0, (x,y) = (2,1), sizeX (W) = 2, sizeY (H) = 4
    InferenceEngine::Blob::Ptr roiBlob =
        make_shared_blob(blob, {0, 0, roi.posY, roi.posX}, {1, 3, roi.posY + roi.sizeY, roi.posX + roi.sizeX});

    // check that BlockingDesc is constructed properly for the ROI blob
    InferenceEngine::SizeVector refDims = {1, 3, 4, 2};
    InferenceEngine::SizeVector refOrder = {0, 1, 2, 3};
    size_t refOffset = 7;
    InferenceEngine::SizeVector refStrides = {90, 30, 5, 1};
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getBlockDims(), refDims);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOrder(), refOrder);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOffsetPadding(), refOffset);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getStrides(), refStrides);
}

TEST_F(BlobTests, makeRangeRoiBlobNhwc) {
    // we create main blob with NHWC layout. We will crop ROI from this blob.
    InferenceEngine::SizeVector dims = {1, 3, 4, 8};  // RGB picture of size (WxH) = 8x4
    InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<uint8_t>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, dims, InferenceEngine::NHWC));
    blob->allocate();

    // create ROI blob based on the already created blob
    InferenceEngine::ROI roi = {0,
                                3,
                                2,
                                5,
                                2};  // cropped picture with: id = 0, (x,y) = (3,2), sizeX (W) = 5, sizeY (H) = 2
    InferenceEngine::Blob::Ptr roiBlob =
        make_shared_blob(blob, {0, 0, roi.posY, roi.posX}, {1, 3, roi.posY + roi.sizeY, roi.posX + roi.sizeX});

    // check that BlockingDesc is constructed properly for the ROI blob
    InferenceEngine::SizeVector refDims = {1, 2, 5, 3};
    InferenceEngine::SizeVector refOrder = {0, 2, 3, 1};
    size_t refOffset = 57;
    InferenceEngine::SizeVector refStrides = {96, 24, 3, 1};
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getBlockDims(), refDims);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOrder(), refOrder);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getOffsetPadding(), refOffset);
    ASSERT_EQ(roiBlob->getTensorDesc().getBlockingDesc().getStrides(), refStrides);
}

TEST_F(BlobTests, makeRangeRoiBlobWrongSize) {
    // we create main blob with NCHW layout. We will crop ROI from this blob.
    InferenceEngine::SizeVector dims = {1, 3, 4, 4};  // RGB picture of size (WxH) = 4x4
    InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<uint8_t>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, dims, InferenceEngine::NCHW));
    blob->allocate();

    // try to create ROI blob with wrong size
    InferenceEngine::ROI roi = {0,
                                1,
                                1,
                                4,
                                4};  // cropped picture with: id = 0, (x,y) = (1,1), sizeX (W) = 4, sizeY (H) = 4
    ASSERT_THROW(make_shared_blob(blob, {0, 0, roi.posY, roi.posX}, {1, 3, roi.posY + roi.sizeY, roi.posX + roi.sizeX}),
                 InferenceEngine::Exception);
}

TEST_F(BlobTests, readRangeRoiBlob) {
    // Create original Blob

    const auto origDesc =
        InferenceEngine::TensorDesc(InferenceEngine::Precision::I32, {1, 3, 4, 8}, InferenceEngine::NCHW);

    const auto origBlob = InferenceEngine::make_shared_blob<int32_t>(origDesc);
    origBlob->allocate();

    // Fill the original Blob

    {
        auto origMemory = origBlob->wmap();
        const auto origPtr = origMemory.as<int32_t*>();
        ASSERT_NE(nullptr, origPtr);

        for (size_t i = 0; i < origBlob->size(); ++i) {
            origPtr[i] = static_cast<int32_t>(i);
        }
    }

    // Create ROI Blob

    const auto roi = InferenceEngine::ROI(0, 4, 2, 4, 2);

    const auto roiBlob = InferenceEngine::as<InferenceEngine::MemoryBlob>(
        origBlob->createROI({0, 0, roi.posY, roi.posX}, {1, 3, roi.posY + roi.sizeY, roi.posX + roi.sizeX}));
    ASSERT_NE(nullptr, roiBlob);

    // Read ROI Blob

    {
        const auto roiOffset = roiBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();

        auto roiMemory = roiBlob->rmap();
        auto roiPtr = roiMemory.as<const int32_t*>();
        ASSERT_NE(nullptr, roiPtr);

        // Blob::rmap returns pointer to the original blob start, we have to add ROI offset manually.
        roiPtr += roiOffset;

        for (size_t i = 0; i < roiBlob->size(); ++i) {
            ASSERT_EQ(roiPtr[i], i + roiOffset);
        }
    }
}

TEST_F(BlobTests, setBiggerShapeOnPreAllocatedMemory) {
    const auto t = ov::make_tensor(ov::element::i64, ov::Shape{2, 6});
    const auto b = ov::tensor_to_blob({t, nullptr});

    const auto origin_ptr = t->data();
    b->setShape({2, 8});

    ASSERT_EQ(b->buffer(), t->data());
    // New allocation, pointer different than origin.
    ASSERT_NE(b->buffer().as<void*>(), origin_ptr);
}

TEST_F(BlobTests, setSmallerShapeOnPreAllocatedMemory) {
    const auto t = ov::make_tensor(ov::element::i64, ov::Shape{2, 6});
    const auto b = ov::tensor_to_blob({t, nullptr});

    const auto origin_ptr = t->data();
    b->setShape({2, 4});

    ASSERT_EQ(b->buffer(), t->data());
    // No new allocation same as origin pointer
    ASSERT_EQ(b->buffer(), origin_ptr);
}
