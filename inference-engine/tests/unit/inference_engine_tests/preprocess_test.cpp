// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_preprocess.hpp>

using namespace std;

class PreProcessTests : public ::testing::Test {
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
    }

public:

};

TEST_F(PreProcessTests, throwsOnSettingNullMeanImage) {
    InferenceEngine::PreProcessInfo info;
    info.init(1);
    ASSERT_THROW(info.setMeanImage(InferenceEngine::Blob::Ptr(nullptr)),
            InferenceEngine::details::InferenceEngineException);

}

TEST_F(PreProcessTests, throwsOnSetting2DMeanImage) {
    InferenceEngine::PreProcessInfo info;
    info.init(1);
    InferenceEngine::Blob::Ptr blob(new InferenceEngine::TBlob<float>(InferenceEngine::Precision::FP32, InferenceEngine::Layout::HW, {1, 1}));
    ASSERT_THROW(info.setMeanImage(blob), InferenceEngine::details::InferenceEngineException);

}

TEST_F(PreProcessTests, throwsOnSettingWrongSizeMeanImage) {
    InferenceEngine::PreProcessInfo info;
    info.init(1);
    InferenceEngine::TBlob<float>::Ptr blob(new InferenceEngine::TBlob<float>(InferenceEngine::Precision::FP32, InferenceEngine::Layout::CHW, { 1, 1, 2 }));
    blob->set({ 1.f, 2.f });
    ASSERT_THROW(info.setMeanImage(blob), InferenceEngine::details::InferenceEngineException);
}

TEST_F(PreProcessTests, noThrowWithCorrectSizeMeanImage) {
    InferenceEngine::PreProcessInfo info;
    info.init(2);
    InferenceEngine::TBlob<float>::Ptr blob(new InferenceEngine::TBlob<float>(InferenceEngine::Precision::FP32, InferenceEngine::Layout::CHW, { 1, 1, 2 }));
    blob->set({ 1.f, 2.f });
    ASSERT_NO_THROW(info.setMeanImage(blob));
}
