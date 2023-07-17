// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_preprocess.hpp>

using namespace std;

IE_SUPPRESS_DEPRECATED_START
using PreProcessTests = ::testing::Test;

TEST_F(PreProcessTests, throwsOnSettingNullMeanImage) {
    InferenceEngine::PreProcessInfo info;
    info.init(1);
    ASSERT_THROW(info.setMeanImage(InferenceEngine::Blob::Ptr(nullptr)), InferenceEngine::Exception);
}

TEST_F(PreProcessTests, throwsOnSetting2DMeanImage) {
    InferenceEngine::PreProcessInfo info;
    info.init(1);
    InferenceEngine::Blob::Ptr blob(
        new InferenceEngine::TBlob<float>({InferenceEngine::Precision::FP32, {1, 1}, InferenceEngine::Layout::HW}));
    ASSERT_THROW(info.setMeanImage(blob), InferenceEngine::Exception);
}

TEST_F(PreProcessTests, throwsOnSettingWrongSizeMeanImage) {
    InferenceEngine::PreProcessInfo info;
    info.init(1);
    InferenceEngine::TBlob<float>::Ptr blob(
        new InferenceEngine::TBlob<float>({InferenceEngine::Precision::FP32, {2, 1, 1}, InferenceEngine::Layout::CHW}));
    blob->allocate();
    ASSERT_THROW(info.setMeanImage(blob), InferenceEngine::Exception);
}

TEST_F(PreProcessTests, noThrowWithCorrectSizeMeanImage) {
    InferenceEngine::PreProcessInfo info;
    info.init(2);
    InferenceEngine::TBlob<float>::Ptr blob(
        new InferenceEngine::TBlob<float>({InferenceEngine::Precision::FP32, {2, 1, 1}, InferenceEngine::Layout::CHW}));
    blob->allocate();
    ASSERT_NO_THROW(info.setMeanImage(blob));
}
