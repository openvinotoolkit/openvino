// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/prior_box.hpp"
#include "base_reference_test.hpp"
#include "openvino/opsets/opset1.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct PriorBoxParams {
    template <class IT>
    PriorBoxParams(const std::vector<float>& min_size,
                   const std::vector<float>& aspect_ratio,
                   const bool scale_all_size,
                   const ov::Shape& layerShapeShape, const ov::Shape& imageShapeShape,
                   const ov::element::Type& iType,
                   const std::vector<IT>& layerShapeValues, const std::vector<IT>& imageShapeValues,
                   const std::vector<float>& oValues,
                   const std::string& testcaseName = "")
        : layerShapeShape(layerShapeShape),
          imageShapeShape(imageShapeShape),
          inType(iType),
          outType(ov::element::Type_t::f32),
          layerShapeData(CreateTensor(iType, layerShapeValues)),
          imageShapeData(CreateTensor(iType, imageShapeValues)),
          refData(CreateTensor(outType, oValues)),
          testcaseName(testcaseName) {
              attrs.min_size = min_size;
              attrs.aspect_ratio = aspect_ratio;
              attrs.scale_all_sizes = scale_all_size;
          }

    ov::op::v0::PriorBox::Attributes attrs;
    ov::Shape layerShapeShape;
    ov::Shape imageShapeShape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::runtime::Tensor layerShapeData;
    ov::runtime::Tensor imageShapeData;
    ov::runtime::Tensor refData;
    std::string testcaseName;
};

class ReferencePriorBoxLayerTest : public testing::TestWithParam<PriorBoxParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<PriorBoxParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "layerShapeShape=" << param.layerShapeShape << "_";
        result << "imageShapeShape=" << param.imageShapeShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PriorBoxParams& params) {
        auto LS = std::make_shared<opset1::Constant>(params.inType, params.layerShapeShape, params.layerShapeData.data());
        auto IS = std::make_shared<opset1::Constant>(params.inType, params.imageShapeShape, params.imageShapeData.data());
        const auto PriorBox = std::make_shared<op::v0::PriorBox>(LS, IS, params.attrs);
        return std::make_shared<ov::Function>(NodeVector {PriorBox}, ParameterVector {});
    }
};

TEST_P(ReferencePriorBoxLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<PriorBoxParams> generatePriorBoxFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<PriorBoxParams> priorBoxParams {
        PriorBoxParams({2.0f}, {1.5f}, false,
                       {2}, {2},
                       IN_ET,
                       std::vector<T>{2, 2},
                       std::vector<T>{10, 10},
                       std::vector<float>{-0.75, -0.75, 1.25, 1.25, -0.974745, -0.566497,  1.47474, 1.0665,
                                          -0.25, -0.75, 1.75, 1.25, -0.474745, -0.566497,  1.97474, 1.0665,
                                          -0.75, -0.25, 1.25, 1.75, -0.974745, -0.0664966, 1.47474, 1.5665,
                                          -0.25, -0.25, 1.75, 1.75, -0.474745, -0.0664966, 1.97474, 1.5665,
                                          0.1,   0.1,   0.1,  0.1,  0.1,       0.1,        0.1,     0.1,
                                          0.1,   0.1,   0.1,  0.1,  0.1,       0.1,        0.1,     0.1,
                                          0.1,   0.1,   0.1,  0.1,  0.1,       0.1,        0.1,     0.1,
                                          0.1,   0.1,   0.1,  0.1,  0.1,       0.1,        0.1,     0.1}),
    };
    return priorBoxParams;
}

std::vector<PriorBoxParams> generatePriorBoxCombinedParams() {
    const std::vector<std::vector<PriorBoxParams>> priorBoxTypeParams {
        generatePriorBoxFloatParams<element::Type_t::i64>(),
        generatePriorBoxFloatParams<element::Type_t::i32>(),
        generatePriorBoxFloatParams<element::Type_t::i16>(),
        generatePriorBoxFloatParams<element::Type_t::i8>(),
        generatePriorBoxFloatParams<element::Type_t::u64>(),
        generatePriorBoxFloatParams<element::Type_t::u32>(),
        generatePriorBoxFloatParams<element::Type_t::u16>(),
        generatePriorBoxFloatParams<element::Type_t::u8>(),
        };
    std::vector<PriorBoxParams> combinedParams;

    for (const auto& params : priorBoxTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_PriorBox_With_Hardcoded_Refs, ReferencePriorBoxLayerTest,
    testing::ValuesIn(generatePriorBoxCombinedParams()), ReferencePriorBoxLayerTest::getTestCaseName);

} // namespace