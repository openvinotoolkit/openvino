// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "inference_engine.hpp"

using namespace std;
using namespace testing;
using namespace InferenceEngine;

class InferenceEngineTests : public ::testing::Test {
public:
	InferenceEngineTests(): output(TensorDesc(Precision::FP32, C))
	{
	}

protected:
    InferenceEngine::TBlob<float> output;
    vector<unsigned> results;
    std::vector<float> values;

    virtual void TearDown() override{
    }

    virtual void SetUp() override {
        values = { 0.3f, 0.1f, 0.01f, 0.9f, 0.99f, 0.12f, 0.001f, 0, 0.999f, 0.0000001f };
        output = TBlob<float>(TensorDesc(Precision::FP32, { 1, 10 }, Layout::NC), &values[0]);
    }

    InferenceEngine::TBlob<float>::Ptr getCopiedTBlob(InferenceEngine::SizeVector size) {
        InferenceEngine::TBlob<float>::Ptr blob(new InferenceEngine::TBlob<float>(TensorDesc(Precision::FP32, size,
                                                                                  TensorDesc::getLayoutByDims(size))));
        blob->allocate();
        const size_t arr_size = 4;
        uint8_t data[arr_size] = { 1, 2, 3, 4 };
        InferenceEngine::copyFromRGB8(&data[0], arr_size, blob.get());
        return blob;
    }
};

TEST_F(InferenceEngineTests, checkZeroInput) {
    InferenceEngine::TBlob<float> output(TensorDesc(Precision::FP32, C));
    output.allocate();
    EXPECT_THROW(InferenceEngine::TopResults(5, output, results), InferenceEngine::details::InferenceEngineException);
}

TEST_F(InferenceEngineTests, testInsertSort) {

    InferenceEngine::TopResults(5, output, results);
    ASSERT_EQ(5, results.size());
    ASSERT_EQ(8, results[0]);
    ASSERT_EQ(4, results[1]);
    ASSERT_EQ(3, results[2]);
    ASSERT_EQ(0, results[3]);
    ASSERT_EQ(5, results[4]);
}

TEST_F(InferenceEngineTests, testInsertSortOverDraft) {

    InferenceEngine::TopResults(15, output, results);
    ASSERT_EQ(10, results.size());
    ASSERT_EQ(8, results[0]);
    ASSERT_EQ(4, results[1]);
    ASSERT_EQ(3, results[2]);
    ASSERT_EQ(0, results[3]);
    ASSERT_EQ(5, results[4]);
    ASSERT_EQ(1, results[5]);
    ASSERT_EQ(2, results[6]);
    ASSERT_EQ(6, results[7]);
    ASSERT_EQ(9, results[8]);
    ASSERT_EQ(7, results[9]);
}

TEST_F(InferenceEngineTests, testThrowsOnCopyToBadBlob) {
    ASSERT_THROW(getCopiedTBlob({ 1, 1, 1 }), InferenceEngine::details::InferenceEngineException);
}

TEST_F(InferenceEngineTests, testThrowsOnCopyToBlobWithBadSize) {
    ASSERT_THROW(getCopiedTBlob({ 1, 1, 1, 1 }), InferenceEngine::details::InferenceEngineException);
}

TEST_F(InferenceEngineTests, canCopyToProperBlob) {
    auto blob = getCopiedTBlob({ 1, 1, 1, 4 });
    ASSERT_EQ(blob->data()[blob->size() - 1], 4);
}
