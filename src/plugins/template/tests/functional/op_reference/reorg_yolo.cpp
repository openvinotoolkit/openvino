// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reorg_yolo.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4244)
#endif

struct ReorgYoloParams {
    template <class IT>
    ReorgYoloParams(const ov::Strides& stride,
                    const ov::Shape& inputShape,
                    const ov::Shape& outputShape,
                    const ov::element::Type& iType,
                    const std::vector<IT>& oValues,
                    const std::string& testcaseName = "")
        : stride(stride),
          inputShape(inputShape),
          inType(iType),
          outType(iType),
          refData(CreateTensor(outputShape, iType, oValues)),
          testcaseName(testcaseName) {
        std::vector<IT> iValues(shape_size(inputShape));
        std::iota(iValues.begin(), iValues.end(), 0);
        inputData = CreateTensor(inputShape, iType, iValues);
    }

    ov::Strides stride;
    ov::Shape inputShape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
    std::string testcaseName;
};

#ifdef _MSC_VER
#    pragma warning(pop)
#endif

class ReferenceReorgYoloLayerTest : public testing::TestWithParam<ReorgYoloParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ReorgYoloParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "inputShape=" << param.inputShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "stride=" << param.stride;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ReorgYoloParams& params) {
        const auto p = std::make_shared<op::v0::Parameter>(params.inType, params.inputShape);
        const auto ReorgYolo = std::make_shared<op::v0::ReorgYolo>(p, params.stride);
        return std::make_shared<ov::Model>(NodeVector{ReorgYolo}, ParameterVector{p});
    }
};

TEST_P(ReferenceReorgYoloLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ReorgYoloParams> generateReorgYoloParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ReorgYoloParams> reorgYoloParams{
        ReorgYoloParams({2},
                        Shape{1, 8, 4, 4},
                        Shape{1, 32, 2, 2},
                        IN_ET,
                        std::vector<T>{0,  2,  4,  6,  16, 18, 20, 22, 32,  34,  36,  38,  48,  50,  52,  54,
                                       64, 66, 68, 70, 80, 82, 84, 86, 96,  98,  100, 102, 112, 114, 116, 118,
                                       1,  3,  5,  7,  17, 19, 21, 23, 33,  35,  37,  39,  49,  51,  53,  55,
                                       65, 67, 69, 71, 81, 83, 85, 87, 97,  99,  101, 103, 113, 115, 117, 119,
                                       8,  10, 12, 14, 24, 26, 28, 30, 40,  42,  44,  46,  56,  58,  60,  62,
                                       72, 74, 76, 78, 88, 90, 92, 94, 104, 106, 108, 110, 120, 122, 124, 126,
                                       9,  11, 13, 15, 25, 27, 29, 31, 41,  43,  45,  47,  57,  59,  61,  63,
                                       73, 75, 77, 79, 89, 91, 93, 95, 105, 107, 109, 111, 121, 123, 125, 127}),
        ReorgYoloParams(
            {3},
            Shape{1, 9, 3, 3},
            Shape{1, 81, 1, 1},
            IN_ET,
            std::vector<T>{0,  3,  6,  27, 30, 33, 54, 57, 60, 1,  4,  7,  28, 31, 34, 55, 58, 61, 2,  5,  8,
                           29, 32, 35, 56, 59, 62, 9,  12, 15, 36, 39, 42, 63, 66, 69, 10, 13, 16, 37, 40, 43,
                           64, 67, 70, 11, 14, 17, 38, 41, 44, 65, 68, 71, 18, 21, 24, 45, 48, 51, 72, 75, 78,
                           19, 22, 25, 46, 49, 52, 73, 76, 79, 20, 23, 26, 47, 50, 53, 74, 77, 80}),
    };
    return reorgYoloParams;
}

std::vector<ReorgYoloParams> generateReorgYoloCombinedParams() {
    const std::vector<std::vector<ReorgYoloParams>> reorgYoloTypeParams{
        generateReorgYoloParams<element::Type_t::f64>(),
        generateReorgYoloParams<element::Type_t::f32>(),
        generateReorgYoloParams<element::Type_t::f16>(),
        generateReorgYoloParams<element::Type_t::bf16>(),
        generateReorgYoloParams<element::Type_t::i64>(),
        generateReorgYoloParams<element::Type_t::i32>(),
        generateReorgYoloParams<element::Type_t::i16>(),
        generateReorgYoloParams<element::Type_t::i8>(),
        generateReorgYoloParams<element::Type_t::u64>(),
        generateReorgYoloParams<element::Type_t::u32>(),
        generateReorgYoloParams<element::Type_t::u16>(),
        generateReorgYoloParams<element::Type_t::u8>(),
    };
    std::vector<ReorgYoloParams> combinedParams;

    for (const auto& params : reorgYoloTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ReorgYolo_With_Hardcoded_Refs,
                         ReferenceReorgYoloLayerTest,
                         testing::ValuesIn(generateReorgYoloCombinedParams()),
                         ReferenceReorgYoloLayerTest::getTestCaseName);

}  // namespace
