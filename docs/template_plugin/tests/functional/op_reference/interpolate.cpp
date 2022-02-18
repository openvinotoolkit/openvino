// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/interpolate.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct InterpolateV1Params {
    template <class IT>
    InterpolateV1Params(const Shape& iShape,
                        const Shape& oShape,
                        const element::Type& iType,
                        const element::Type& oType,
                        const std::vector<IT>& iValues,
                        const std::vector<IT>& oValues,
                        const std::shared_ptr<op::v0::Constant>& outShapeInput,
                        const AxisSet& axes,
                        const std::string& mode,
                        const bool& align_corners,
                        const bool& antialias,
                        const std::vector<size_t>& pads_begin,
                        const std::vector<size_t>& pads_end)
        : inShape(iShape),
          outShape(oShape),
          inType(iType),
          outType(oType),
          inData(CreateTensor(iType, iValues)),
          outData(CreateTensor(oType, oValues)),
          outShapeInput(outShapeInput) {
        attrs.axes = axes;
        attrs.mode = mode;
        attrs.align_corners = align_corners;
        attrs.antialias = antialias;
        attrs.pads_begin = pads_begin;
        attrs.pads_end = pads_end;
    }

    Shape inShape;
    Shape outShape;
    element::Type inType;
    element::Type outType;
    runtime::Tensor inData;
    runtime::Tensor outData;
    std::shared_ptr<op::v0::Constant> outShapeInput;
    op::v0::Interpolate::Attributes attrs;
};

struct InterpolateV4Params {
    template <class IT>
    InterpolateV4Params(const Shape& iShape,
                        const Shape& oShape,
                        const element::Type& iType,
                        const element::Type& oType,
                        const std::vector<IT>& iValues,
                        const std::vector<IT>& oValues,
                        const std::vector<size_t> outShapeInput,
                        const element::Type& outShapeInputType,
                        const std::vector<float>& scales,
                        const op::v4::Interpolate::InterpolateAttrs attrs)
        : inShape(iShape),
          outShape(oShape),
          inType(iType),
          outType(oType),
          inData(CreateTensor(iType, iValues)),
          outData(CreateTensor(oType, oValues)),
          outShapeInput(outShapeInput),
          outShapeInputType(outShapeInputType),
          scales(scales),
          attrs(attrs) {}
    Shape inShape;
    Shape outShape;
    element::Type inType;
    element::Type outType;
    runtime::Tensor inData;
    runtime::Tensor outData;
    std::vector<size_t> outShapeInput;
    element::Type outShapeInputType;
    std::vector<float> scales;
    op::v4::Interpolate::InterpolateAttrs attrs;
};

class ReferenceInterpolateV1LayerTest : public testing::TestWithParam<InterpolateV1Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.inShape,
                                  params.outShape,
                                  params.inType,
                                  params.outType,
                                  params.outShapeInput,
                                  params.attrs);
        inputData = {params.inData};
        refOutData = {params.outData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateV1Params>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iShape=" << param.inShape << "_";
        result << "oShape=" << param.outShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const Shape& output_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& output_type,
                                                 const std::shared_ptr<op::v0::Constant>& output_shape_input,
                                                 op::v0::Interpolate::Attributes& attrs) {
        const auto input = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto interpolate = std::make_shared<op::v0::Interpolate>(input, output_shape_input, attrs);
        return std::make_shared<Model>(NodeVector{interpolate}, ParameterVector{input});
    }
};

class ReferenceInterpolateV4LayerTest : public testing::TestWithParam<InterpolateV4Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.inShape,
                                  params.outShape,
                                  params.inType,
                                  params.outType,
                                  params.outShapeInput,
                                  params.outShapeInputType,
                                  params.scales,
                                  params.attrs);
        inputData = {params.inData};
        refOutData = {params.outData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateV4Params>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iShape=" << param.inShape << "_";
        result << "oShape=" << param.outShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const Shape& output_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& output_type,
                                                 const std::vector<size_t> outShapeInput,
                                                 const element::Type& outShapeInputType,
                                                 const std::vector<float>& scales,
                                                 op::v4::Interpolate::InterpolateAttrs& attrs) {
        const auto node_input = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto node_output_shape_input = op::v0::Constant::create(outShapeInputType, outShapeInput, output_shape);
        const auto node_scales = op::v0::Constant::create(element::Type_t::f32, {scales.size()}, scales);
        auto interpolate =
            std::make_shared<op::v4::Interpolate>(node_input, node_output_shape_input, node_scales, attrs);
        return std::make_shared<Model>(NodeVector{interpolate}, ParameterVector{node_input});
    }
};

TEST_P(ReferenceInterpolateV1LayerTest, InterpolateWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceInterpolateV4LayerTest, InterpolateWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<InterpolateV1Params> generateParamsForInterpolateV1() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<InterpolateV1Params> params{
        InterpolateV1Params(ov::Shape{1, 1, 2, 4},
                            ov::Shape{1, 1, 1, 2},
                            IN_ET,
                            IN_ET,
                            std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                            std::vector<T>{1, 3},
                            op::v0::Constant::create(element::i64, {4}, {1, 1, 1, 2}),
                            AxisSet{0, 1, 2, 3},
                            "nearest",
                            false,
                            false,
                            std::vector<size_t>{0, 0, 0, 0},
                            std::vector<size_t>{0, 0, 0, 0})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<InterpolateV4Params> generateParamsForInterpolateV4() {
    using T = typename element_type_traits<IN_ET>::value_type;
    using InterpolateAttrs = op::v4::Interpolate::InterpolateAttrs;
    using InterpolateMode = op::v4::Interpolate::InterpolateMode;
    using ShapeCalcMode = op::v4::Interpolate::ShapeCalcMode;
    using TransformMode = op::v4::Interpolate::CoordinateTransformMode;
    using NearestMode = op::v4::Interpolate::NearestMode;

    std::vector<InterpolateV4Params> params{InterpolateV4Params(ov::Shape{1, 1, 2, 4},
                                                                ov::Shape{1, 1, 1, 2},
                                                                IN_ET,
                                                                IN_ET,
                                                                std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                                                                std::vector<T>{1, 3},
                                                                {4},
                                                                element::i64,
                                                                {1.0},
                                                                InterpolateAttrs{InterpolateMode::NEAREST,
                                                                                 ShapeCalcMode::SIZES,
                                                                                 std::vector<size_t>{0, 0, 0, 0},
                                                                                 std::vector<size_t>{0, 0, 0, 0},
                                                                                 TransformMode::HALF_PIXEL,
                                                                                 NearestMode::ROUND_PREFER_FLOOR,
                                                                                 false,
                                                                                 -0.75})};
    return params;
}

std::vector<InterpolateV1Params> generateCombinedParamsForInterpolateV1() {
    const std::vector<std::vector<InterpolateV1Params>> allTypeParams{
        generateParamsForInterpolateV1<element::Type_t::f64>(),
        generateParamsForInterpolateV1<element::Type_t::f32>(),
        generateParamsForInterpolateV1<element::Type_t::f16>(),
        generateParamsForInterpolateV1<element::Type_t::bf16>()};

    std::vector<InterpolateV1Params> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<InterpolateV4Params> generateCombinedParamsForInterpolateV4() {
    const std::vector<std::vector<InterpolateV4Params>> allTypeParams{
        generateParamsForInterpolateV4<element::Type_t::f32>(),
        generateParamsForInterpolateV4<element::Type_t::f16>(),
        generateParamsForInterpolateV4<element::Type_t::bf16>(),
        generateParamsForInterpolateV4<element::Type_t::i8>(),
        generateParamsForInterpolateV4<element::Type_t::u8>()};

    std::vector<InterpolateV4Params> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_V1_With_Hardcoded_Refs,
                         ReferenceInterpolateV1LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForInterpolateV1()),
                         ReferenceInterpolateV1LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_V4_With_Hardcoded_Refs,
                         ReferenceInterpolateV4LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForInterpolateV4()),
                         ReferenceInterpolateV4LayerTest::getTestCaseName);

}  // namespace
