// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/extractimagepatches.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ExtractImagePatchesParams {
    reference_tests::Tensor data;
    Shape sizes;
    Strides strides;
    Shape rates;
    op::PadType autoPad;
    reference_tests::Tensor expectedResult;
    std::string testcaseName;
};

struct Builder : ParamsBuilder<ExtractImagePatchesParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, data);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, sizes);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, strides);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, rates);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, autoPad);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expectedResult);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, testcaseName);
};

class ReferenceExtractImagePatchesTest : public testing::TestWithParam<ExtractImagePatchesParams>,
                                         public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateModel(params);
        inputData = {params.data.data};
        refOutData = {params.expectedResult.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ExtractImagePatchesParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.data.type;
        result << "_dShape=" << param.data.shape;
        result << "_sizes=" << param.sizes;
        result << "_strides=" << param.strides;
        result << "_rates=" << param.rates;
        result << "_autoPad=" << param.autoPad;
        result << "_eType=" << param.expectedResult.type;
        result << "_eShape=" << param.expectedResult.shape;
        if (param.testcaseName != "") {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateModel(const ExtractImagePatchesParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.data.type, params.data.shape);
        const auto extrace_image_patches = std::make_shared<op::v3::ExtractImagePatches>(data,
                                                                                         params.sizes,
                                                                                         params.strides,
                                                                                         params.rates,
                                                                                         params.autoPad);
        const auto f = std::make_shared<Model>(extrace_image_patches, ParameterVector{data});
        return f;
    }
};

TEST_P(ReferenceExtractImagePatchesTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ExtractImagePatchesParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ExtractImagePatchesParams> params{
        Builder{}
            .data({ET,
                   {1, 1, 10, 10},
                   std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                  41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                  61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                  81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100}})
            .sizes({3, 3})
            .strides({5, 5})
            .rates({1, 1})
            .autoPad(op::PadType::VALID)
            .expectedResult({ET, {1, 9, 2, 2}, std::vector<T>{1,  6,  51, 56, 2,  7,  52, 57, 3,  8,  53, 58,
                                                              11, 16, 61, 66, 12, 17, 62, 67, 13, 18, 63, 68,
                                                              21, 26, 71, 76, 22, 27, 72, 77, 23, 28, 73, 78}}),

        Builder{}
            .data({ET,
                   {1, 1, 10, 10},
                   std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                  41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                  61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                  81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100}})
            .sizes({4, 4})
            .strides({8, 8})
            .rates({1, 1})
            .autoPad(op::PadType::VALID)
            .expectedResult(
                {ET, {1, 16, 1, 1}, std::vector<T>{1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34}}),

        Builder{}
            .data({ET,
                   {1, 1, 10, 10},
                   std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                  41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                  61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                  81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100}})
            .sizes({4, 4})
            .strides({9, 9})
            .rates({1, 1})
            .autoPad(op::PadType::SAME_UPPER)
            .expectedResult(
                {ET, {1, 16, 2, 2}, std::vector<T>{0, 0,  0, 89, 0,  0,  81, 90,  0,  0, 82, 0, 0,  0, 83, 0,
                                                   0, 9,  0, 99, 1,  10, 91, 100, 2,  0, 92, 0, 3,  0, 93, 0,
                                                   0, 19, 0, 0,  11, 20, 0,  0,   12, 0, 0,  0, 13, 0, 0,  0,
                                                   0, 29, 0, 0,  21, 30, 0,  0,   22, 0, 0,  0, 23, 0, 0,  0}}),

        Builder{}
            .data({ET,
                   {1, 1, 10, 10},
                   std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                  41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                  61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                  81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100}})
            .sizes({3, 3})
            .strides({5, 5})
            .rates({2, 2})
            .autoPad(op::PadType::VALID)
            .expectedResult({ET, {1, 9, 2, 2}, std::vector<T>{1,  6,  51, 56, 3,  8,  53, 58, 5,  10, 55, 60,
                                                              21, 26, 71, 76, 23, 28, 73, 78, 25, 30, 75, 80,
                                                              41, 46, 91, 96, 43, 48, 93, 98, 45, 50, 95, 100}}),

        Builder{}
            .data({ET, {1, 2, 5, 5}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                                                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                                    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50}})
            .sizes({2, 2})
            .strides({3, 3})
            .rates({1, 1})
            .autoPad(op::PadType::VALID)
            .expectedResult(
                {ET, {1, 8, 2, 2}, std::vector<T>{1, 4, 16, 19, 26, 29, 41, 44, 2, 5,  17, 20, 27, 30, 42, 45,
                                                  6, 9, 21, 24, 31, 34, 46, 49, 7, 10, 22, 25, 32, 35, 47, 50}}),
    };
    return params;
}

std::vector<ExtractImagePatchesParams> generateCombinedParams() {
    const std::vector<std::vector<ExtractImagePatchesParams>> generatedParams{
        generateParams<element::Type_t::i8>(),
        generateParams<element::Type_t::i16>(),
        generateParams<element::Type_t::i32>(),
        generateParams<element::Type_t::i64>(),
        generateParams<element::Type_t::u8>(),
        generateParams<element::Type_t::u16>(),
        generateParams<element::Type_t::u32>(),
        generateParams<element::Type_t::u64>(),
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<ExtractImagePatchesParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ExtractImagePatches_With_Hardcoded_Refs,
                         ReferenceExtractImagePatchesTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceExtractImagePatchesTest::getTestCaseName);
}  // namespace
