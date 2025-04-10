// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prior_box.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct PriorBoxParams {
    template <class IT>
    PriorBoxParams(const std::vector<float>& min_size,
                   const std::vector<float>& aspect_ratio,
                   const bool scale_all_size,
                   const ov::element::Type& iType,
                   const std::vector<IT>& layerShapeValues,
                   const std::vector<IT>& imageShapeValues,
                   const Shape& output_shape,
                   const std::vector<float>& oValues,
                   const std::string& testcaseName = "")
        : inType(iType),
          outType(ov::element::Type_t::f32),
          layerShapeData(CreateTensor(Shape{2}, iType, layerShapeValues)),
          imageShapeData(CreateTensor(Shape{2}, iType, imageShapeValues)),
          refData(CreateTensor(output_shape, outType, oValues)),
          testcaseName(testcaseName) {
        attrs.min_size = min_size;
        attrs.aspect_ratio = aspect_ratio;
        attrs.scale_all_sizes = scale_all_size;
    }

    ov::op::v0::PriorBox::Attributes attrs;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor layerShapeData;
    ov::Tensor imageShapeData;
    ov::Tensor refData;
    std::string testcaseName;
};

struct PriorBoxV8Params {
    template <class IT>
    PriorBoxV8Params(const std::vector<float>& min_size,
                     const std::vector<float>& max_size,
                     const std::vector<float>& aspect_ratio,
                     const bool scale_all_size,
                     const bool min_max_aspect_ratios_order,
                     const ov::element::Type& iType,
                     const std::vector<IT>& layerShapeValues,
                     const std::vector<IT>& imageShapeValues,
                     const Shape& output_shape,
                     const std::vector<float>& oValues,
                     const std::string& testcaseName = "")
        : inType(iType),
          outType(ov::element::Type_t::f32),
          layerShapeData(CreateTensor(Shape{2}, iType, layerShapeValues)),
          imageShapeData(CreateTensor(Shape{2}, iType, imageShapeValues)),
          refData(CreateTensor(output_shape, outType, oValues)),
          testcaseName(testcaseName) {
        attrs.min_size = min_size;
        attrs.max_size = max_size;
        attrs.aspect_ratio = aspect_ratio;
        attrs.scale_all_sizes = scale_all_size;
        attrs.min_max_aspect_ratios_order = min_max_aspect_ratios_order;
    }

    ov::op::v8::PriorBox::Attributes attrs;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor layerShapeData;
    ov::Tensor imageShapeData;
    ov::Tensor refData;
    std::string testcaseName;
};

class ReferencePriorBoxLayerTest : public testing::TestWithParam<PriorBoxParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<PriorBoxParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        if (!param.testcaseName.empty())
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PriorBoxParams& params) {
        const auto LS = std::make_shared<op::v0::Constant>(params.layerShapeData);
        const auto IS = std::make_shared<op::v0::Constant>(params.imageShapeData);
        const auto PriorBox = std::make_shared<op::v0::PriorBox>(LS, IS, params.attrs);
        return std::make_shared<ov::Model>(NodeVector{PriorBox}, ParameterVector{});
    }
};

class ReferencePriorBoxV8LayerTest : public testing::TestWithParam<PriorBoxV8Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<PriorBoxV8Params>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        if (!param.testcaseName.empty())
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PriorBoxV8Params& params) {
        const auto LS = std::make_shared<op::v0::Constant>(params.layerShapeData);
        const auto IS = std::make_shared<op::v0::Constant>(params.imageShapeData);
        const auto PriorBoxV8 = std::make_shared<op::v8::PriorBox>(LS, IS, params.attrs);
        return std::make_shared<ov::Model>(NodeVector{PriorBoxV8}, ParameterVector{});
    }
};

TEST_P(ReferencePriorBoxLayerTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferencePriorBoxV8LayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<PriorBoxParams> generatePriorBoxFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<PriorBoxParams> priorBoxParams{
        PriorBoxParams({2.0f},
                       {1.5f},
                       false,
                       IN_ET,
                       std::vector<T>{2, 2},
                       std::vector<T>{10, 10},
                       Shape{2, 32},
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

template <element::Type_t IN_ET>
std::vector<PriorBoxV8Params> generatePriorBoxV8FloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<PriorBoxV8Params> priorBoxV8Params{
        PriorBoxV8Params(
            {2.0f},
            {5.0f},
            {1.5f},
            true,
            false,
            IN_ET,
            std::vector<T>{2, 2},
            std::vector<T>{10, 10},
            Shape{2, 48},
            std::vector<float>{
                0.15, 0.15, 0.35, 0.35, 0.127526, 0.16835, 0.372474, 0.33165, 0.0918861, 0.0918861, 0.408114, 0.408114,
                0.65, 0.15, 0.85, 0.35, 0.627526, 0.16835, 0.872474, 0.33165, 0.591886,  0.0918861, 0.908114, 0.408114,
                0.15, 0.65, 0.35, 0.85, 0.127526, 0.66835, 0.372474, 0.83165, 0.0918861, 0.591886,  0.408114, 0.908114,
                0.65, 0.65, 0.85, 0.85, 0.627526, 0.66835, 0.872474, 0.83165, 0.591886,  0.591886,  0.908114, 0.908114,
                0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1,
                0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1,
                0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1,
                0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1}),
    };
    return priorBoxV8Params;
}

std::vector<PriorBoxParams> generatePriorBoxCombinedParams() {
    const std::vector<std::vector<PriorBoxParams>> priorBoxTypeParams{
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

std::vector<PriorBoxV8Params> generatePriorBoxV8CombinedParams() {
    const std::vector<std::vector<PriorBoxV8Params>> priorBoxV8TypeParams{
        generatePriorBoxV8FloatParams<element::Type_t::i64>(),
        generatePriorBoxV8FloatParams<element::Type_t::i32>(),
        generatePriorBoxV8FloatParams<element::Type_t::i16>(),
        generatePriorBoxV8FloatParams<element::Type_t::i8>(),
        generatePriorBoxV8FloatParams<element::Type_t::u64>(),
        generatePriorBoxV8FloatParams<element::Type_t::u32>(),
        generatePriorBoxV8FloatParams<element::Type_t::u16>(),
        generatePriorBoxV8FloatParams<element::Type_t::u8>(),
    };
    std::vector<PriorBoxV8Params> combinedParams;

    for (const auto& params : priorBoxV8TypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_PriorBox_With_Hardcoded_Refs,
                         ReferencePriorBoxLayerTest,
                         testing::ValuesIn(generatePriorBoxCombinedParams()),
                         ReferencePriorBoxLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PriorBoxV8_With_Hardcoded_Refs,
                         ReferencePriorBoxV8LayerTest,
                         testing::ValuesIn(generatePriorBoxV8CombinedParams()),
                         ReferencePriorBoxV8LayerTest::getTestCaseName);
}  // namespace
