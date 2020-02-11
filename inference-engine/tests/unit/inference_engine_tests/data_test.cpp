// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>

#include <ie_data.h>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class DataTests : public ::testing::Test {
protected:
    const std::string data_name = "test_data_name";
    const Precision precision = Precision::FP32;

    const SizeVector notEmptyDims = {1, 1, 1, 1};
    const SizeVector emptyDims = {};
    const size_t batchSize = 1;

    class BlockingDescTest : public BlockingDesc {
    public:
        BlockingDescTest(const SizeVector &blocked_dims, const SizeVector &order) : BlockingDesc(blocked_dims, order) {}

        void fillDescTest(const SizeVector &blocked_dims, const SizeVector &order) {
            fillDesc(blocked_dims, order);
        }
    };
};

TEST_F(DataTests, canSetEmptyDimsForDataDefault) {
    Data data(data_name, precision);
    ASSERT_NO_THROW(data.setDims(emptyDims));
    ASSERT_FALSE(data.isInitialized());
}

TEST_F(DataTests, canSetEmptyDimsForDataBlocked) {
    Data data(data_name, precision, BLOCKED);
    ASSERT_NO_THROW(data.setDims(emptyDims));
}

TEST_F(DataTests, canSetNotEmptyDimsForDataBlocked) {
    Data data(data_name, precision, BLOCKED);
    ASSERT_NO_THROW(data.setDims(notEmptyDims));
}

TEST_F(DataTests, canSetNotEmptyDimsForDataNCHW) {
    Data data(data_name, precision, NCHW);
    ASSERT_NO_THROW(data.setDims(notEmptyDims));
    ASSERT_TRUE(data.isInitialized());
}

TEST_F(DataTests, canSetEmptyDimsForTensorDescNCHW) {
    TensorDesc desc(precision, emptyDims, NCHW);
    ASSERT_NO_THROW(desc.setDims(emptyDims));
}

TEST_F(DataTests, canSetEmptyDimsForTensorDescBlocked) {
    TensorDesc desc(precision, emptyDims, BLOCKED);
    ASSERT_NO_THROW(desc.setDims(emptyDims));
}

TEST_F(DataTests, canSetNotEmptyDimsForTensorDescBlocked) {
    TensorDesc desc(precision, notEmptyDims, BLOCKED);
    ASSERT_NO_THROW(desc.setDims(notEmptyDims));
}

TEST_F(DataTests, canSetEmptyDimsForBlockingDescOrder) {
    ASSERT_NO_THROW(BlockingDesc(emptyDims, emptyDims));
}

TEST_F(DataTests, throwOnFillDescByEmptyDimsForBlockingDesc) {
    BlockingDescTest desc(emptyDims, emptyDims);
    ASSERT_THROW(desc.fillDescTest(emptyDims, emptyDims), InferenceEngineException);
}

TEST_F(DataTests, throwOnSetEmptyDimsForBlockingDescBlocked) {
    ASSERT_NO_THROW(BlockingDesc(emptyDims, BLOCKED));
}

TEST_F(DataTests, throwOnSetEmptyDimsForBlockingDescNCHW) {
    ASSERT_NO_THROW(BlockingDesc(emptyDims, NCHW));
}

TEST_F(DataTests, canSetNotEmptyDimsForBlockingDescBlocked) {
    ASSERT_NO_THROW(BlockingDesc(notEmptyDims, BLOCKED));
}

TEST_F(DataTests, canSetNotEmptyDimsForBlockingDescNCHW) {
    ASSERT_NO_THROW(BlockingDesc(notEmptyDims, NCHW));
}

TEST_F(DataTests, setPrecision) {
    Data data(data_name, { Precision::FP32, emptyDims, Layout::NCHW });

    EXPECT_EQ(Precision::FP32, data.getPrecision());
    EXPECT_EQ(Precision::FP32, data.getTensorDesc().getPrecision());

    data.setPrecision(Precision::FP16);
    EXPECT_EQ(Precision::FP16, data.getPrecision());
    EXPECT_EQ(Precision::FP16, data.getTensorDesc().getPrecision());
}
