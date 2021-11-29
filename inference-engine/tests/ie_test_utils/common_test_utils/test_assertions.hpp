// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include <ie_data.h>
#include <ie_input_info.hpp>
#include <ie_blob.h>
#include <ie_common.h>
#include <ie_preprocess.hpp>

inline bool strContains(const std::string & str, const std::string & substr) {
    return str.find(substr) != std::string::npos;
}

inline bool strDoesnotContain(const std::string & str, const std::string & substr) {
    return !strContains(str, substr);
}

#define ASSERT_STR_CONTAINS(str, substr) \
    ASSERT_PRED2(&strContains, str, substr)

#define ASSERT_STR_DOES_NOT_CONTAIN(str, substr) \
    ASSERT_PRED2(&strDoesnotContain, str, substr)

#define EXPECT_STR_CONTAINS(str, substr) \
    EXPECT_PRED2(&strContains, str, substr)

#define ASSERT_BLOB_EQ(lhs, rhs) \
    compare_blob(lhs, rhs)

#define ASSERT_DIMS_EQ(lhs, rhs) \
    compare_dims(lhs, rhs)

#define ASSERT_DATA_EQ(lhs, rhs) \
    compare_data(lhs, rhs)

#define ASSERT_PREPROCESS_CHANNEL_EQ(lhs, rhs) \
    compare_preprocess(lhs, rhs)

#define ASSERT_PREPROCESS_INFO_EQ(lhs, rhs) \
    compare_preprocess_info(lhs, rhs)

#define ASSERT_OUTPUTS_INFO_EQ(lhs, rhs) \
    compare_outputs_info(lhs, rhs)

#define ASSERT_INPUTS_INFO_EQ(lhs, rhs) \
    compare_inputs_info(lhs, rhs)

#define ASSERT_STRINGEQ(lhs, rhs) \
    compare_cpp_strings(lhs, rhs)

inline void compare_blob(InferenceEngine::Blob::Ptr lhs, InferenceEngine::Blob::Ptr rhs) {
    ASSERT_EQ(lhs.get(), rhs.get());
    //TODO: add blob specific comparison for general case
}

inline void compare_dims(const InferenceEngine::SizeVector & lhs, const InferenceEngine::SizeVector & rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());
    for (size_t i = 0; i < lhs.size(); i++) {
        ASSERT_EQ(lhs[i], rhs[i]);
    }
}

inline void compare_data(const InferenceEngine::Data & lhs, const InferenceEngine::Data & rhs) {
    ASSERT_DIMS_EQ(lhs.getDims(), rhs.getDims());
    ASSERT_STREQ(lhs.getName().c_str(), rhs.getName().c_str());
    ASSERT_EQ(lhs.getPrecision(), rhs.getPrecision());
}

inline void compare_preprocess(const InferenceEngine::PreProcessChannel & lhs, const InferenceEngine::PreProcessChannel & rhs) {
    ASSERT_FLOAT_EQ(lhs.meanValue, rhs.meanValue);
    ASSERT_FLOAT_EQ(lhs.stdScale, rhs.stdScale);
    ASSERT_BLOB_EQ(lhs.meanData, rhs.meanData);
}

inline void compare_preprocess_info(const InferenceEngine::PreProcessInfo & lhs, const InferenceEngine::PreProcessInfo & rhs) {
    ASSERT_EQ(lhs.getMeanVariant(), rhs.getMeanVariant());
    ASSERT_EQ(lhs.getNumberOfChannels(), rhs.getNumberOfChannels());
    for (size_t i = 0; i < lhs.getNumberOfChannels(); i++) {
        ASSERT_PREPROCESS_CHANNEL_EQ(*lhs[i].get(), *rhs[i].get());
    }
}

inline void compare_outputs_info(const InferenceEngine::OutputsDataMap & lhs, const InferenceEngine::OutputsDataMap & rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());
    auto i = lhs.begin();
    auto j = rhs.begin();

    for (size_t k =0; k != lhs.size(); k++, i++, j++) {
        ASSERT_STREQ(i->first.c_str(), j->first.c_str());
        ASSERT_DATA_EQ(*i->second.get(), *j->second.get());
    }
}

inline void compare_inputs_info(const InferenceEngine::InputsDataMap & lhs, const InferenceEngine::InputsDataMap & rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());
    auto i = lhs.begin();
    auto j = rhs.begin();

    for (size_t k = 0; k != lhs.size(); k++, i++, j++) {
        ASSERT_STREQ(i->first.c_str(), j->first.c_str());
        ASSERT_DIMS_EQ(i->second->getTensorDesc().getDims(), j->second->getTensorDesc().getDims());
        ASSERT_PREPROCESS_INFO_EQ(i->second->getPreProcess(), j->second->getPreProcess());
        ASSERT_DATA_EQ(*i->second->getInputData().get(), *j->second->getInputData().get());
    }
}

inline void compare_cpp_strings(const std::string & lhs, const std::string &rhs) {
    ASSERT_STREQ(lhs.c_str(), rhs.c_str());
}
