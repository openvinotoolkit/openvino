// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <ie_core.hpp>
#include <legacy/cnn_network_impl.hpp>
#include <legacy/ie_layers_property.hpp>
#include <precision_utils.h>
#include <xml_net_builder.hpp>
#include <tests_common.hpp>

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 9) && !defined(__clang__)
# define IE_GCC_4_8
#endif

#ifndef IE_GCC_4_8
# include <regex>
# define REPLACE_WITH_STR(SRC, PATTERN, STR) SRC = std::regex_replace(SRC, std::regex(PATTERN), STR)
# define FIND_STR(SRC, PATTERN) std::regex_search(SRC, std::regex(PATTERN))
#else
# error "Cannot implement regex"
# define REPLACE_WITH_STR(SRC, PATTERN, STR)
# define FIND_STR(SRC, PATTERN)
#endif

#define REPLACE_WITH_NUM(SRC, PATTERN, NUM) REPLACE_WITH_STR(SRC, PATTERN, to_string_c_locale(NUM))

#define REPLACE_WITH_NUM_VECTOR(SRC, PATTERN, NUMS) \
    { std::string result; \
        if (NUMS.size() > 0u) { \
            result += to_string_c_locale(NUMS[0]); \
            for (size_t i = 1u; i < NUMS.size(); i++) { \
                    result += "," + to_string_c_locale(NUMS[i]); \
            } \
        } \
    REPLACE_WITH_STR(SRC, PATTERN, result); }

#define REPLACE_WITH_NUM_VECTOR_REVERSE(SRC, PATTERN, NUMS) \
    { std::string result; \
        auto nums_size = NUMS.size(); \
        if (nums_size > 0u) { \
            result += to_string_c_locale(NUMS[nums_size - 1]); \
            for (size_t i = 2u; i <= nums_size; i++) { \
                    result += "," + to_string_c_locale(NUMS[nums_size - i]); \
            } \
        } \
    REPLACE_WITH_STR(SRC, PATTERN, result); }

#define REMOVE_LINE(SRC, PATTERN) REPLACE_WITH_STR(SRC, PATTERN, "")

#define PRETTY_PARAM(name, type)                                                            \
    class name                                                                              \
    {                                                                                       \
    public:                                                                                 \
        typedef type param_type;                                                            \
        name ( param_type arg = param_type ()) : val_(arg) {}                               \
        operator param_type () const {return val_;}                                         \
    private:                                                                                \
        param_type val_;                                                                    \
    };                                                                                      \
    static inline void PrintTo(name param, ::std::ostream* os)                              \
    {                                                                                       \
        *os << #name ": " << ::testing::PrintToString((name::param_type)(param));           \
    }

template<int Version = 3>
inline InferenceEngine::CNNNetwork
buildSingleLayerNetworkCommon(const std::string &layerType,
                              const CommonTestUtils::InOutShapes &inOutShapes,
                              std::map<std::string, std::string> *params,
                              const std::string &layerDataName = "data",
                              const InferenceEngine::Precision &precision = InferenceEngine::Precision::FP32,
                              size_t weightsSize = 0,
                              size_t biasesSize = 0,
                              const InferenceEngine::TBlob<uint8_t>::Ptr &weights = nullptr) {
    std::string precisionStr = precision.name();
    auto netBuilder = CommonTestUtils::XmlNetBuilder<Version>::buildNetworkWithOneInput("Mock", inOutShapes.inDims[0],
                                                                                precisionStr);
    size_t inputsNumber = inOutShapes.inDims.size();
    for (int i = 1; i < inputsNumber; i++) {
        netBuilder.addInputLayer(precisionStr, inOutShapes.inDims[i]);
    }
    netBuilder.addLayer(layerType, precisionStr, params, inOutShapes, weightsSize, biasesSize, layerDataName);
    std::string testContent;
    if (inputsNumber > 1) {
        auto edgeBuilder = netBuilder.havingEdges();
        for (size_t i = 0lu; i < inputsNumber; i++) {
            edgeBuilder.connect(i, inputsNumber);
        }
        testContent = edgeBuilder.finish();
    } else {
        testContent = netBuilder.finish();
    }

    InferenceEngine::Core ie;
    return ie.ReadNetwork(testContent, weights);
}

void GenRandomDataCommon(InferenceEngine::Blob::Ptr blob);

class BufferWrapper {
    InferenceEngine::Precision precision;
    InferenceEngine::ie_fp16 *fp16_ptr;
    float *fp32_ptr;
    int32_t *i32_ptr;
    uint8_t *u8_ptr;
public:
    explicit BufferWrapper(const InferenceEngine::Blob::Ptr &blob);

    BufferWrapper(const InferenceEngine::Blob::Ptr &blob, InferenceEngine::Precision precision);

    float operator[](size_t index);

    void insert(size_t index, float value);
};

void CompareCommon(const InferenceEngine::Blob::Ptr &actual,
                   const InferenceEngine::Blob::Ptr &expected,
                   const std::function<void(size_t, size_t)> &errorUpdater);

void CompareCommonExact(const InferenceEngine::Blob::Ptr &actual,
                        const InferenceEngine::Blob::Ptr &expected);

void CompareCommonAbsolute(const InferenceEngine::Blob::Ptr &actual,
                           const InferenceEngine::Blob::Ptr &expected,
                           float tolerance);

void CompareCommonRelative(const InferenceEngine::Blob::Ptr &actual,
                           const InferenceEngine::Blob::Ptr &expected,
                           float tolerance);

void CompareCommonCombined(const InferenceEngine::Blob::Ptr &actual,
                           const InferenceEngine::Blob::Ptr &expected,
                           float tolerance);

void CompareCommonWithNorm(const InferenceEngine::Blob::Ptr &actual,
                           const InferenceEngine::Blob::Ptr &expected,
                           float maxDiff);

void fill_data_common(BufferWrapper &data, size_t size, size_t duty_ratio = 10);
