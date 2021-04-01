// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include "memory/gna_memory.hpp"

using namespace GNAPluginNS::memory;

class GNAMemoryTest : public ::testing::Test {

 protected:
    GNAMemory<std::allocator<uint8_t>> mem;

    void SetUp() override  {
    }
};

TEST_F(GNAMemoryTest, canStoreActualBlob){
    float input [] = {1,2,3};
    float* pFuture = nullptr;
    size_t len = sizeof(input);

    mem.push_ptr(&pFuture, input, len);
    mem.commit();

    ASSERT_NE(pFuture, nullptr);
    ASSERT_NE(pFuture, input);
    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
}

TEST_F(GNAMemoryTest, canStore2Blobs) {
    float input [] = {1,2,3,4};
    float* pFuture = nullptr;
    float* pFuture2 = nullptr;

    mem.push_ptr(&pFuture, input, 3*4);
    mem.push_ptr(&pFuture2, input+1, 3*4);
    mem.commit();

    ASSERT_NE(pFuture, input);
    ASSERT_NE(pFuture2, input);
    ASSERT_EQ(pFuture + 3, pFuture2);

    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
    ASSERT_EQ(pFuture[3], 2);
    ASSERT_EQ(pFuture[4], 3);
    ASSERT_EQ(pFuture[5], 4);
}

TEST_F(GNAMemoryTest, canStoreBlobsALIGNED) {
    float input [] = {1,2,3,4,5,6,7,8};
    float* pFuture = nullptr;

    mem.push_ptr(&pFuture, input, 3*4, 8);
    mem.commit();

    ASSERT_EQ(16 , mem.getTotalBytes());

    ASSERT_NE(pFuture, input);
    ASSERT_NE(pFuture, nullptr);

    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
    //least probability for next element to be equal if not copied
    ASSERT_NE(pFuture[3], 4);
}

TEST_F(GNAMemoryTest, canStore2BlobsALIGNED) {
    float input [] = {1,2,3,4,5,6,7,8};
    float* pFuture = nullptr;
    float* pFuture2 = nullptr;

    mem.push_ptr(&pFuture, input, 3*4, 8);
    mem.push_ptr(&pFuture2, input, 3*4, 16);
    mem.commit();

    ASSERT_EQ(32 , mem.getTotalBytes());

    ASSERT_NE(pFuture, nullptr);

    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
    //least probability for next element to be equal if not copied
    ASSERT_EQ(pFuture[4], 1);
    ASSERT_EQ(pFuture[5], 2);
    ASSERT_EQ(pFuture[6], 3);

}

TEST_F(GNAMemoryTest, canReserveData) {

    float* pFuture = nullptr;
    mem.reserve_ptr(&pFuture, 3*4);
    mem.commit();

    ASSERT_NE(pFuture, nullptr);
}

TEST_F(GNAMemoryTest, canReserveDataByVoid) {
    mem.reserve_ptr(nullptr, 3*4);
    ASSERT_NO_THROW(mem.commit());
}


TEST_F(GNAMemoryTest, canReserveAndPushData) {

    float input[] = {1, 2, 3};
    float *pFuture = nullptr;
    float* pFuture2 = nullptr;
    size_t len = sizeof(input) ;

    mem.push_ptr(&pFuture, input, len);
    mem.reserve_ptr(&pFuture2, 3*4);
    mem.commit();

    ASSERT_NE(pFuture, nullptr);
    ASSERT_NE(pFuture2, nullptr);
    ASSERT_NE(pFuture, input);
    ASSERT_NE(pFuture2, pFuture);

    pFuture2[0] = -1;
    pFuture2[1] = -1;
    pFuture2[2] = -1;

    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
}

TEST_F(GNAMemoryTest, canBindAndResolve) {

    float input[] = {1, 2, 3};
    float *pFuture = nullptr;
    float *pFuture2 = nullptr;
    float *pFuture3 = nullptr;
    size_t len = sizeof(input);

    mem.bind_ptr(&pFuture3, &pFuture);
    mem.push_ptr(&pFuture, input, len);
    mem.bind_ptr(&pFuture2, &pFuture);

    mem.commit();

    ASSERT_NE(pFuture, input);
    ASSERT_NE(pFuture2, nullptr);
    ASSERT_EQ(pFuture2, pFuture);
    ASSERT_EQ(pFuture3, pFuture);

    ASSERT_EQ(pFuture2[0], 1);
    ASSERT_EQ(pFuture2[1], 2);
    ASSERT_EQ(pFuture2[2], 3);
}

TEST_F(GNAMemoryTest, canBindTransitevlyAndResolve) {

    float input[] = {1, 2, 3};
    float *pFuture = nullptr;
    float *pFuture3 = nullptr;
    float *pFuture4 = nullptr;
    size_t len = sizeof(input);

    mem.bind_ptr(&pFuture4, &pFuture3);
    mem.bind_ptr(&pFuture3, &pFuture);
    mem.push_ptr(&pFuture, input, len);

    mem.commit();

    ASSERT_NE(pFuture, input);
    ASSERT_EQ(pFuture3, pFuture);
    ASSERT_EQ(pFuture4, pFuture);

    ASSERT_NE(pFuture4, nullptr);

    ASSERT_EQ(pFuture4[0], 1);
    ASSERT_EQ(pFuture4[1], 2);
    ASSERT_EQ(pFuture4[2], 3);
}

TEST_F(GNAMemoryTest, canBindTransitevlyWithOffsetsAndResolve) {

    float input[] = {1, 2, 3};
    float *pFuture = nullptr;
    float *pFuture3 = nullptr;
    float *pFuture4 = nullptr;
    size_t len = sizeof(input);

    mem.bind_ptr(&pFuture4, &pFuture3, 4);
    mem.bind_ptr(&pFuture3, &pFuture, 4);
    mem.push_ptr(&pFuture, input, len);

    mem.commit();

    ASSERT_NE(pFuture, input);
    ASSERT_EQ(pFuture3, pFuture + 1);
    ASSERT_EQ(pFuture4, pFuture + 2);

    ASSERT_NE(pFuture, nullptr);

    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
}

TEST_F(GNAMemoryTest, canBindWithOffsetAndResolve) {

    float input[] = {1, 2, 3};
    float *pFuture = nullptr;
    float *pFuture2 = nullptr;
    float *pFuture3 = nullptr;
    size_t len = sizeof(input);

    mem.bind_ptr(&pFuture3, &pFuture, 4);
    mem.push_ptr(&pFuture, input, len);
    mem.bind_ptr(&pFuture2, &pFuture);

    mem.commit();

    ASSERT_NE(pFuture, input);
    ASSERT_NE(pFuture2, nullptr);
    ASSERT_EQ(pFuture2, pFuture);
    ASSERT_NE(pFuture3, nullptr);
    ASSERT_EQ(pFuture3, pFuture + 1);

    ASSERT_EQ(pFuture2[0], 1);
    ASSERT_EQ(pFuture2[1], 2);
    ASSERT_EQ(pFuture2[2], 3);
    ASSERT_EQ(pFuture3[0], 2);
}


TEST_F(GNAMemoryTest, canPushLocal) {

    float* pFuture = (float*)&pFuture;

    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
        mem.push_local_ptr(pFuture, &*input.begin(), 4 * 4, 1);
    }

    //poison stack
    mem.commit();

    ASSERT_FLOAT_EQ(pFuture[0], 1);
    ASSERT_FLOAT_EQ(pFuture[1], 2);
    ASSERT_FLOAT_EQ(pFuture[2], 3);
    ASSERT_FLOAT_EQ(pFuture[3], 4);
}

TEST_F(GNAMemoryTest, canPushValue) {

    float* pFuture = (float*)&pFuture;
    float* pFuture2 = (float*)&pFuture2;

    {
        mem.push_value(pFuture, 3.f,  2);
        mem.push_value(pFuture2, 13.f, 2);
    }

    mem.commit();

    ASSERT_FLOAT_EQ(pFuture[0], 3);
    ASSERT_FLOAT_EQ(pFuture[1], 3);
    ASSERT_FLOAT_EQ(pFuture[2], 13);
    ASSERT_FLOAT_EQ(pFuture[3], 13);
}

TEST_F(GNAMemoryTest, canPushReadOnlyValue) {

    float* pFuture = (float*)&pFuture;
    float* pFuture2 = (float*)&pFuture2;

    {
        mem.push_value(pFuture, 3.f,  2);
        mem.readonly().push_value(pFuture2, 13.f, 2);
    }

    mem.commit();

    ASSERT_FLOAT_EQ(pFuture[0], 3);
    ASSERT_FLOAT_EQ(pFuture[1], 3);
    ASSERT_FLOAT_EQ(pFuture[2], 13);
    ASSERT_FLOAT_EQ(pFuture[3], 13);
}

TEST_F(GNAMemoryTest, canCalculateReadWriteSectionSize) {

    mem.push_value(nullptr, 3.f,  2);
    mem.readonly().push_value(nullptr, 13.f, 2);
    mem.commit();

    ASSERT_EQ(mem.getTotalBytes(), 4 * sizeof(float));
    ASSERT_EQ(mem.getRWBytes(), 2 * sizeof(float));
}

TEST_F(GNAMemoryTest, canCalculateReadWriteSectionSizeWithAlignment) {

    GNAMemory<std::allocator<uint8_t>> memAligned(64);

    memAligned.push_value(nullptr, 3.f,  2);
    memAligned.readonly().push_value(nullptr, 13.f, 2);
    memAligned.commit();

    ASSERT_EQ(memAligned.getTotalBytes(), 128);
    ASSERT_EQ(memAligned.getRWBytes(), 64);
}

TEST_F(GNAMemoryTest, canSetUpReadWriteSectionPtr) {

    float* pFuture2 = (float*)&pFuture2;
    float* pFuture1 = (float*)&pFuture1;
    float* pFuture3 = (float*)&pFuture3;


    mem.readonly().push_value(pFuture1, 3.f,  2);
    mem.push_value(pFuture2, 13.f, 3);
    mem.readonly().push_value(pFuture3, 32.f,  4);
    mem.commit();

    ASSERT_EQ(mem.getTotalBytes(), (2+3+4) * sizeof(float));
    ASSERT_EQ(mem.getRWBytes(), 3 * sizeof(float));

    ASSERT_LT(&pFuture2[0], &pFuture1[0]);
    ASSERT_LT(&pFuture1[0], &pFuture3[0]);

    ASSERT_FLOAT_EQ(pFuture1[0], 3.f);
    ASSERT_FLOAT_EQ(pFuture1[1], 3.f);

    ASSERT_FLOAT_EQ(pFuture2[0], 13.f);
    ASSERT_FLOAT_EQ(pFuture2[1], 13.f);
    ASSERT_FLOAT_EQ(pFuture2[2], 13.f);

    ASSERT_FLOAT_EQ(pFuture3[0], 32.f);
    ASSERT_FLOAT_EQ(pFuture3[1], 32.f);
    ASSERT_FLOAT_EQ(pFuture3[2], 32.f);
    ASSERT_FLOAT_EQ(pFuture3[3], 32.f);
}


TEST_F(GNAMemoryTest, canUpdateSizeOfPushRequestWithBindRequest) {
    float input[]  = {1, 2, 3};

    float *pFuture = nullptr;
    float *pFuture2 = nullptr;
    float *pFuture3 = nullptr;

    size_t len = sizeof(input);

    mem.push_ptr(&pFuture, input, len);
    mem.bind_ptr(&pFuture2, &pFuture, len, len);
    mem.bind_ptr(&pFuture3, &pFuture2, 2 * len, len);

    mem.commit();

    ASSERT_EQ(mem.getTotalBytes(), 4 * len);
    ASSERT_NE(pFuture, nullptr);
    ASSERT_EQ(pFuture2, pFuture + 3);
    ASSERT_EQ(pFuture3, pFuture + 9);

    ASSERT_FLOAT_EQ(pFuture[0], 1);
    ASSERT_FLOAT_EQ(pFuture[1], 2);
    ASSERT_FLOAT_EQ(pFuture[2], 3);
    ASSERT_FLOAT_EQ(pFuture[3], 0);
    ASSERT_FLOAT_EQ(pFuture[4], 0);
    ASSERT_FLOAT_EQ(pFuture[5], 0);
    ASSERT_FLOAT_EQ(pFuture[6], 0);
    ASSERT_FLOAT_EQ(pFuture[7], 0);
    ASSERT_FLOAT_EQ(pFuture[8], 0);
}

TEST_F(GNAMemoryTest, canUpdateSizeOfPushRequestWithBindRequestWhenPush) {
    float input[]  = {1, 2, 3};
    float input2[]  = {6, 7, 8};

    float *pFutureInput2 = nullptr;
    float *pFuture = nullptr;
    float *pFuture2 = nullptr;

    size_t len = sizeof(input);

    mem.push_ptr(&pFuture, input, len);
    mem.bind_ptr(&pFuture2, &pFuture, len, len);
    mem.push_ptr(&pFutureInput2, input2, len);

    mem.commit();

    ASSERT_EQ(mem.getTotalBytes(), 3 * len);
    ASSERT_NE(pFuture, nullptr);
    ASSERT_NE(pFutureInput2, nullptr);
    ASSERT_EQ(pFuture2, pFuture + 3);

    ASSERT_FLOAT_EQ(pFuture[0], 1);
    ASSERT_FLOAT_EQ(pFuture[1], 2);
    ASSERT_FLOAT_EQ(pFuture[2], 3);
    ASSERT_FLOAT_EQ(pFuture[3], 0);
    ASSERT_FLOAT_EQ(pFuture[4], 0);

    ASSERT_FLOAT_EQ(pFutureInput2[0], 6);
    ASSERT_FLOAT_EQ(pFutureInput2[1], 7);
    ASSERT_FLOAT_EQ(pFutureInput2[2], 8);
}

TEST_F(GNAMemoryTest, canUpdateSizeOfPushRequestWithBindRequestWhenAlloc) {
    float input[]  = {1, 2, 3};

    float *pFutureInput = nullptr;
    float *pFuture = nullptr;
    float *pFuture2 = nullptr;

    size_t len = sizeof(input);

    mem.reserve_ptr(&pFuture, len);
    mem.bind_ptr(&pFuture2, &pFuture, len, len);
    mem.push_ptr(&pFutureInput, input, len);

    mem.commit();

    ASSERT_EQ(mem.getTotalBytes(), 3 * len);
    ASSERT_NE(pFuture, nullptr);
    ASSERT_NE(pFutureInput, nullptr);
    ASSERT_EQ(pFuture2, pFuture + 3);

    ASSERT_FLOAT_EQ(pFuture[0], 0);
    ASSERT_FLOAT_EQ(pFuture[1], 0);
    ASSERT_FLOAT_EQ(pFuture[2], 0);
    ASSERT_FLOAT_EQ(pFuture[3], 0);
    ASSERT_FLOAT_EQ(pFuture[4], 0);

    ASSERT_FLOAT_EQ(pFutureInput[0], 1);
    ASSERT_FLOAT_EQ(pFutureInput[1], 2);
    ASSERT_FLOAT_EQ(pFutureInput[2], 3);
}