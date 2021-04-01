// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_compound_blob.h>
#include <gtest/gtest.h>
#include <random>
#include <chrono>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;

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
class I420BlobTests : public CompoundBlobTests {};

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
        MemoryBlob::Ptr mb = as<MemoryBlob>(blobs[i]);
        auto lm = mb->rwmap();
        lm.as<uint8_t*>()[0] = static_cast<uint8_t>(i + MAGIC_NUMBER);
    }

    _test_blob = make_shared_blob<CompoundBlob>(blobs);

    verifyCompoundBlob(_test_blob, blobs);

    CompoundBlob::Ptr compound_blob = as<CompoundBlob>(_test_blob);
    EXPECT_EQ(blobs.size(), compound_blob->size());
    for (size_t i = 0; i < compound_blob->size(); ++i) {
        auto blob = compound_blob->getBlob(i);
        ASSERT_NE(nullptr, blob);
        MemoryBlob::Ptr mb = as<MemoryBlob>(blob);
        ASSERT_NE(nullptr, mb);
        auto lm = mb->rwmap();
        EXPECT_EQ(static_cast<uint8_t>(i + MAGIC_NUMBER), lm.as<uint8_t *>()[0]);
    }
}

TEST_F(CompoundBlobTests, compoundBlobHoldsReferencesToBlobs) {
    // Create a blob with HW layout and pass it to a compound blob to check that the compound blob
    // holds references to the blob and not a copy of it
    MemoryBlob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1}, HW));
    blob->allocate();
    // here is quite self to dereference address since LockedMemory would be destroyed only after assignemnt
    blob->rwmap().as<uint8_t*>()[0] = 12;
    _test_blob = make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>({blob}));

    verifyCompoundBlob(_test_blob);

    CompoundBlob::Ptr compound_blob = as<CompoundBlob>(_test_blob);
    Blob::Ptr b0 = compound_blob->getBlob(0);
    MemoryBlob::CPtr mb0 = as<MemoryBlob>(b0);
    EXPECT_EQ(12, mb0->rmap().as<const uint8_t *>()[0]);
    blob->rwmap().as<uint8_t*>()[0] = 34;
    EXPECT_EQ(34, mb0->rmap().as<const uint8_t *>()[0]);
}

TEST_F(CompoundBlobTests, compoundBlobHoldsValidDataWhenUnderlyingBlobIsDestroyed) {
    // Create a scoped blob with HW layout, pass it to compound, and destroy the original scoped
    // blob. Check that the compound blob, which holds a reference to the destroyed blob, still has
    // a valid object
    static constexpr const uint8_t stored_value = 123;
    {
        MemoryBlob::Ptr blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1}, HW));
        blob->allocate();
        blob->rwmap().as<uint8_t*>()[0] = stored_value;
        _test_blob = make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>({blob}));
    }

    verifyCompoundBlob(_test_blob);
    CompoundBlob::Ptr compound_blob = as<CompoundBlob>(_test_blob);
    ASSERT_NE(nullptr, compound_blob->getBlob(0));
    MemoryBlob::CPtr mb0 = as<MemoryBlob>(compound_blob->getBlob(0));
    ASSERT_NE(nullptr, mb0);
    EXPECT_EQ(stored_value, mb0->rmap().as<const uint8_t *>()[0]);
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
    NV12Blob::Ptr nv12_blob = make_shared_blob<NV12Blob>(
        make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 8}, NHWC)),
        make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 3, 4}, NHWC)));
    verifyCompoundBlob(nv12_blob);
}

TEST_F(I420BlobTests, canCreateI420BlobFromThreePlanes) {
    Blob::Ptr y_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 8}, NHWC));
    Blob::Ptr u_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 3, 4}, NHWC));
    Blob::Ptr v_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 3, 4}, NHWC));
    I420Blob::Ptr i420_blob = make_shared_blob<I420Blob>(y_blob, u_blob, v_blob);
    verifyCompoundBlob(i420_blob, {y_blob, u_blob, v_blob});
    EXPECT_EQ(y_blob, i420_blob->y());
    EXPECT_EQ(u_blob, i420_blob->u());
    EXPECT_EQ(v_blob, i420_blob->v());
}

TEST_F(I420BlobTests, canCreateI420BlobFromThreeMovedPlanes) {
    I420Blob::Ptr i420_blob = make_shared_blob<I420Blob>(
        make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 8}, NHWC)),
        make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 3, 4}, NHWC)),
        make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 3, 4}, NHWC)));
    verifyCompoundBlob(i420_blob);
}

TEST_F(I420BlobTests, cannotCreateI420BlobFromNullptrBlobs) {
    Blob::Ptr valid = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 4, 4}, NHWC));
    EXPECT_THROW(make_shared_blob<I420Blob>(valid, nullptr, nullptr), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<I420Blob>(nullptr, valid, nullptr), InferenceEngine::details::InferenceEngineException);
}

TEST_F(I420BlobTests, cannotCreateI420BlobFromCompoundBlobs) {
    Blob::Ptr y_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 8}, NHWC));
    Blob::Ptr u_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 3, 4}, NHWC));
    Blob::Ptr v_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 3, 4}, NHWC));

    auto make_cblob = [](Blob::Ptr const& b){
        return make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>({b}));
    };

    auto c_y_blob = make_cblob(y_blob);
    auto c_u_blob = make_cblob(u_blob);
    auto c_v_blob = make_cblob(v_blob);
    using ie_exception_t = InferenceEngine::details::InferenceEngineException;

    EXPECT_THROW(make_shared_blob<I420Blob>(c_y_blob, u_blob,   v_blob), ie_exception_t);
    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob,   c_u_blob, v_blob), ie_exception_t);
    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob,   u_blob,   c_v_blob), ie_exception_t);
}

TEST_F(I420BlobTests, cannotCreateI420BlobFromPlanesWithDifferentElementSize) {
    Blob::Ptr y_blob_u8    = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 4, 4}, NHWC));
    Blob::Ptr u_blob_float = make_shared_blob<float>(TensorDesc(Precision::FP32, {1, 1, 2, 2}, NHWC));
    Blob::Ptr v_blob_float = make_shared_blob<float>(TensorDesc(Precision::FP32, {1, 1, 2, 2}, NHWC));

    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob_u8, u_blob_float, v_blob_float), InferenceEngine::details::InferenceEngineException);
}

TEST_F(I420BlobTests, cannotCreateI420BlobFromPlanesWithNonU8Precision) {
    Blob::Ptr y_blob_float = make_shared_blob<float>(TensorDesc(Precision::FP32, {1, 1, 4, 4}, NHWC));
    Blob::Ptr u_blob_float = make_shared_blob<float>(TensorDesc(Precision::FP32, {1, 1, 2, 2}, NHWC));
    Blob::Ptr v_blob_float = make_shared_blob<float>(TensorDesc(Precision::FP32, {1, 1, 2, 2}, NHWC));

    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob_float, u_blob_float, v_blob_float), InferenceEngine::details::InferenceEngineException);
}

TEST_F(I420BlobTests, cannotCreateI420BlobFromPlanesWithInconsistentBatchSize) {
    Blob::Ptr y_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 8}, NHWC));
    Blob::Ptr u_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {2, 1, 3, 4}, NHWC));
    Blob::Ptr v_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 3, 4}, NHWC));
    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob, u_blob, v_blob), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob, v_blob, u_blob), InferenceEngine::details::InferenceEngineException);
}

TEST_F(I420BlobTests, cannotCreateI420BlobFromPlanesWithWrongChannelNumber) {
    Blob::Ptr y_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 8}, NHWC));
    Blob::Ptr u_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, 3, 4}, NHWC));
    Blob::Ptr v_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 3, 4}, NHWC));
    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob, u_blob, v_blob), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob, v_blob, u_blob), InferenceEngine::details::InferenceEngineException);
}

TEST_F(I420BlobTests, cannotCreateI420BlobFromPlanesWithWrongWidthRatio) {
    Blob::Ptr y_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 8}, NHWC));
    Blob::Ptr u_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 3, 2}, NHWC));
    Blob::Ptr v_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 3, 4}, NHWC));

    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob, u_blob, v_blob), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob, v_blob, u_blob), InferenceEngine::details::InferenceEngineException);
}

TEST_F(I420BlobTests, cannotCreateI420BlobFromPlanesWithWrongHeightRatio) {
    Blob::Ptr y_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 6, 8}, NHWC));
    Blob::Ptr u_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 2, 4}, NHWC));
    Blob::Ptr v_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, 3, 4}, NHWC));

    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob, u_blob, v_blob), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(make_shared_blob<I420Blob>(y_blob, v_blob, u_blob), InferenceEngine::details::InferenceEngineException);
}


