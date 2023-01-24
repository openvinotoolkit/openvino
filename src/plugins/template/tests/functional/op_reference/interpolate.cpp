// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/interpolate.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;
using namespace reference_tests;

namespace {
namespace type_tests {

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
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inData};
        refOutData = {params.outData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateV1Params>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "iShape=" << param.inShape << "_";
        result << "oShape=" << param.outShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const InterpolateV1Params& params) {
        const auto input = std::make_shared<op::v0::Parameter>(params.inType, params.inShape);
        const auto interpolate = std::make_shared<op::v0::Interpolate>(input, params.outShapeInput, params.attrs);
        return std::make_shared<Model>(NodeVector{interpolate}, ParameterVector{input});
    }
};

class ReferenceInterpolateV4LayerTest : public testing::TestWithParam<InterpolateV4Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inData};
        refOutData = {params.outData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateV4Params>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "iShape=" << param.inShape << "_";
        result << "oShape=" << param.outShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const InterpolateV4Params& params) {
        const auto node_input = std::make_shared<op::v0::Parameter>(params.inType, params.inShape);
        const auto node_output_shape_input =
            op::v0::Constant::create(params.outShapeInputType, params.outShapeInput, params.outShape);
        const auto node_scales = op::v0::Constant::create(element::Type_t::f32, {params.scales.size()}, params.scales);
        auto interpolate =
            std::make_shared<op::v4::Interpolate>(node_input, node_output_shape_input, node_scales, params.attrs);
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
        std::move(params.begin(), params.end(), std::back_inserter(combinedParams));
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

    for (auto& params : allTypeParams) {
        std::move(params.begin(), params.end(), std::back_inserter(combinedParams));
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

}  // namespace type_tests

namespace attribute_tests {
using InterpolateAttrs = op::v4::Interpolate::InterpolateAttrs;
using InterpolateMode = op::v4::Interpolate::InterpolateMode;
using ShapeCalcMode = op::v4::Interpolate::ShapeCalcMode;
using CoordinateTransformMode = op::v4::Interpolate::CoordinateTransformMode;
using TransformMode = op::v4::Interpolate::CoordinateTransformMode;
using NearestMode = op::v4::Interpolate::NearestMode;

struct InterpolateV4TestParams {
    std::string test_name;
    Shape input_data_shape;
    std::vector<int64_t> spatial_shape_data;
    Shape output_shape;
    std::vector<float> scales_data;
    std::vector<int64_t> axes_data;
    InterpolateAttrs attrs;
    std::vector<float> input_data;
    std::vector<float> expected_results;
};

std::vector<InterpolateV4TestParams> generateParamsForInterpolate_v4_cubic() {
    const auto input_data_shape = Shape{1, 1, 4, 4};
    const std::vector<float> input_data =
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    const std::vector<size_t> zero_pads{0, 0, 0, 0};
    // clang-format off
    return {
        {   "cubic.resize_downsample_scales_cubic",
            input_data_shape,
            {3, 3},
            Shape{1, 1, 3, 3},
            {0.8f, 0.8f},
            {2, 3},
            {   InterpolateMode::CUBIC,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            input_data,
            {1.47119141, 2.78125, 4.08251953, 6.71142578, 8.02148438, 9.32275391, 11.91650391, 13.2265625, 14.52783203},
        },
        {   "cubic.resize_downsample_sizes_cubic",
            input_data_shape,
            {3, 3},
            Shape{1, 1, 3, 3},
            {0.75f, 0.75f},
            {2, 3},
            {   InterpolateMode::CUBIC,
                ShapeCalcMode::SIZES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            input_data,
            {1.63078704f, 3.00462963f, 4.37847222f, 7.12615741f, 8.5f, 9.87384259f, 12.62152778f, 13.99537037f,
             15.36921296f},
        },
        {   "cubic.resize_upsample_scales_cubic",
            input_data_shape,
            {8, 8},
            Shape{1, 1, 8, 8},
            {2.0f, 2.0f},
            {2, 3},
            {   InterpolateMode::CUBIC,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            input_data,
            {0.47265625f,  0.76953125f,  1.24609375f,  1.875f,    2.28125f,  2.91015625f,  3.38671875f,  3.68359375f,
             1.66015625f,  1.95703125f,  2.43359375f,  3.0625f,   3.46875f,  4.09765625f,  4.57421875f,  4.87109375f,
             3.56640625f,  3.86328125f,  4.33984375f,  4.96875f,  5.375f,    6.00390625f,  6.48046875f,  6.77734375f,
             6.08203125f,  6.37890625f,  6.85546875f,  7.484375f, 7.890625f, 8.51953125f,  8.99609375f,  9.29296875f,
             7.70703125f,  8.00390625f,  8.48046875f,  9.109375f, 9.515625f, 10.14453125f, 10.62109375f, 10.91796875f,
             10.22265625f, 10.51953125f, 10.99609375f, 11.625f,   12.03125f, 12.66015625f, 13.13671875f, 13.43359375f,
             12.12890625f, 12.42578125f, 12.90234375f, 13.53125f, 13.9375f,  14.56640625f, 15.04296875f, 15.33984375f,
             13.31640625f, 13.61328125f, 14.08984375f, 14.71875f, 15.125f,   15.75390625f, 16.23046875f, 16.52734375f},
        },
        {   "cubic.resize_upsample_scales_cubic_asymmetric",
            input_data_shape,
            {8, 8},
            Shape{1, 1, 8, 8},
            {2.0f, 2.0f},
            {2, 3},
            {   InterpolateMode::CUBIC,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::ASYMMETRIC,
                NearestMode::ROUND_PREFER_FLOOR},
            input_data,
            {1.0f,    1.40625f,  2.0f,    2.5f,      3.0f,    3.59375f,  4.0f,    4.09375f,  2.625f,  3.03125f,
             3.625f,  4.125f,    4.625f,  5.21875f,  5.625f,  5.71875f,  5.0f,    5.40625f,  6.0f,    6.5f,
             7.0f,    7.59375f,  8.0f,    8.09375f,  7.0f,    7.40625f,  8.0f,    8.5f,      9.0f,    9.59375f,
             10.0f,   10.09375f, 9.0f,    9.40625f,  10.0f,   10.5f,     11.0f,   11.59375f, 12.0f,   12.09375f,
             11.375f, 11.78125f, 12.375f, 12.875f,   13.375f, 13.96875f, 14.375f, 14.46875f, 13.0f,   13.40625f,
             14.0f,   14.5f,     15.0f,   15.59375f, 16.0f,   16.09375f, 13.375f, 13.78125f, 14.375f, 14.875f,
             15.375f, 15.96875f, 16.375f, 16.46875f},
        },
        {   "cubic.resize_upsample_sizes_cubic",
            input_data_shape,
            {9, 10},
            Shape{1, 1, 9, 10},
            {2.25f, 2.5f},
            {2, 3},
            {   InterpolateMode::CUBIC,
                ShapeCalcMode::SIZES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            input_data,
            {0.45507922,  0.64057922,  0.97157922,  1.42257922,  1.90732922,  2.22332922,  2.70807922,  3.15907922,
             3.49007922,  3.67557922,  1.39437963,  1.57987963,  1.91087963,  2.36187963,  2.84662963,  3.16262963,
             3.64737963,  4.09837963,  4.42937963,  4.61487963,  2.95130693,  3.13680693,  3.46780693,  3.91880693,
             4.40355693,  4.71955693,  5.20430693,  5.65530693,  5.98630693,  6.17180693,  5.20525069,  5.39075069,
             5.72175069,  6.17275069,  6.65750069,  6.97350069,  7.45825069,  7.90925069,  8.24025069,  8.42575069,
             6.88975,     7.07525,     7.40625,     7.85725,     8.342,       8.658,       9.14275,     9.59375,
             9.92475,     10.11025,    8.57424931,  8.75974931,  9.09074931,  9.54174931,  10.02649931, 10.34249931,
             10.82724931, 11.27824931, 11.60924931, 11.79474931, 10.82819307, 11.01369307, 11.34469307, 11.79569307,
             12.28044307, 12.59644307, 13.08119307, 13.53219307, 13.86319307, 14.04869307, 12.38512037, 12.57062037,
             12.90162037, 13.35262037, 13.83737037, 14.15337037, 14.63812037, 15.08912037, 15.42012037, 15.60562037,
             13.32442078, 13.50992078, 13.84092078, 14.29192078, 14.77667078, 15.09267078, 15.57742078, 16.02842078,
             16.35942078, 16.54492078},
        },
        {   "cubic.resize_downsample_scales_cubic_align_corners",
            input_data_shape,
            {3, 3},
            Shape{1, 1, 3, 3},
            {0.8f, 0.8f},
            {2, 3},
            {   InterpolateMode::CUBIC,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::ALIGN_CORNERS,
                NearestMode::ROUND_PREFER_FLOOR},
            input_data,
            {1.0f, 2.5f, 4.0f, 7.0f, 8.5f, 10.0f, 13.0f, 14.5f, 16.0f},
        },
        {   "cubic.resize_upsample_scales_cubic_align_corners",
            input_data_shape,
            {8, 8},
            Shape{1, 1, 8, 8},
            {2.0f, 2.0f},
            {2, 3},
            {   InterpolateMode::CUBIC,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::ALIGN_CORNERS,
                NearestMode::ROUND_PREFER_FLOOR},
            input_data,
            {1.0,         1.34110787,  1.80029155,  2.32944606,  2.67055394,  3.19970845,  3.65889213,  4.0,
             2.36443149,  2.70553936,  3.16472303,  3.69387755,  4.03498542,  4.56413994,  5.02332362,  5.36443149,
             4.20116618,  4.54227405,  5.00145773,  5.53061224,  5.87172012,  6.40087464,  6.86005831,  7.20116618,
             6.31778426,  6.65889213,  7.1180758,   7.64723032,  7.98833819,  8.51749271,  8.97667638,  9.31778426,
             7.68221574,  8.02332362,  8.48250729,  9.01166181,  9.35276968,  9.8819242,   10.34110787, 10.68221574,
             9.79883382,  10.13994169, 10.59912536, 11.12827988, 11.46938776, 11.99854227, 12.45772595, 12.79883382,
             11.63556851, 11.97667638, 12.43586006, 12.96501458, 13.30612245, 13.83527697, 14.29446064, 14.63556851,
             13.0,        13.34110787, 13.80029155, 14.32944606, 14.67055394, 15.19970845, 15.65889213, 16.0}
        }
    };
    // clang-format on
}

std::vector<InterpolateV4TestParams> generateParamsForInterpolate_v4_nearest() {
    const std::vector<size_t> zero_pads{0, 0, 0, 0};
    // clang-format off
    return {
        {   "nearest.resize_downsample_scales_nearest",
            Shape{1, 1, 2, 4},
            {1, 2},
            Shape{1, 1, 1, 2},
            {0.6f, 0.6f},
            {2, 3},
            {   InterpolateMode::NEAREST,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
            {1.0f, 3.0f},
        },
        {   "nearest.resize_downsample_sizes_nearest",
            Shape{1, 1, 2, 4},
            {1, 2},
            Shape{1, 1, 1, 2},
            {0.5f, 0.5f},
            {2, 3},
            {   InterpolateMode::NEAREST,
                ShapeCalcMode::SIZES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
            {1.0f, 3.0f},
        },
        {   "nearest.resize_downsample_sizes_nearest_tf_half_pixel_for_nn",
            Shape{1, 1, 4, 4},
            {3, 2},
            Shape{1, 1, 3, 2},
            {0.75, 0.5},
            {2, 3},
            {   InterpolateMode::NEAREST,
                ShapeCalcMode::SIZES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
            {6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f},
        },
        {   "nearest.resize_upsample_scales_nearest",
            Shape{1, 1, 2, 2},
            {4, 6},
            Shape{1, 1, 4, 6},
            {2.0f, 3.0f},
            {2, 3},
            {   InterpolateMode::NEAREST,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
             3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f},
        },
        {   "nearest.resize_upsample_sizes_nearest",
            Shape{1, 1, 2, 2},
            {7, 8},
            Shape{1, 1, 7, 8},
            {3.5f, 4.0f},
            {2, 3},
            {   InterpolateMode::NEAREST,
                ShapeCalcMode::SIZES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f,
             2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f,
             2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f,
             3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f},
        },
        {   "nearest.resize_upsample_sizes_nearest_ceil_half_pixel",
            Shape{1, 1, 4, 4},
            {8, 8},
            Shape{1, 1, 8, 8},
            {2.0f, 2.0f},
            {2, 3},
            {   InterpolateMode::NEAREST,
                ShapeCalcMode::SIZES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::CEIL},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
            {1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  4.0f,  4.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,
             8.0f,  8.0f,  8.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,  8.0f,  8.0f,  9.0f,  10.0f,
             10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 12.0f, 9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f,
             12.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f,
             15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f},
        },
        {   "nearest.resize_upsample_sizes_nearest_floor_align_corners",
            Shape{1, 1, 4, 4},
            {8, 8},
            Shape{1, 1, 8, 8},
            {2.0f, 2.0f},
            {2, 3},
            {   InterpolateMode::NEAREST,
                ShapeCalcMode::SIZES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::ALIGN_CORNERS,
                NearestMode::FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
            {1.0f, 1.0f, 1.0f, 2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  1.0f,  1.0f,  1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,
             1.0f, 1.0f, 1.0f, 2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  5.0f,  5.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,
             5.0f, 5.0f, 5.0f, 6.0f,  6.0f,  7.0f,  7.0f,  8.0f,  9.0f,  9.0f,  9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f,
             9.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 13.0f, 13.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f},
        },
        {   "nearest.resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric",
            Shape{1, 1, 4, 4},
            {8, 8},
            Shape{1, 1, 8, 8},
            {2.0f, 2.0f},
            {2, 3},
            {   InterpolateMode::NEAREST,
                ShapeCalcMode::SIZES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::ASYMMETRIC,
                NearestMode::ROUND_PREFER_CEIL},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
            {1.0,  2.0,  2.0,  3.0,  3.0,  4.0,  4.0,  4.0,  5.0,  6.0,  6.0,  7.0,  7.0,  8.0,  8.0,  8.0,
             5.0,  6.0,  6.0,  7.0,  7.0,  8.0,  8.0,  8.0,  9.0,  10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 12.0,
             9.0,  10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 12.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0,
             13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0},
        },
    };
    // clang-format on
}

std::vector<InterpolateV4TestParams> generateParamsForInterpolate_v4_linear_onnx() {
    const std::vector<size_t> zero_pads{0, 0, 0, 0};
    // clang-format off
    return {
        {   "linear_onnx.resize_downsample_scales_linear",
            Shape{1, 1, 2, 4},
            {1, 2},
            Shape{1, 1, 1, 2},
            {0.6f, 0.6f},
            {2, 3},
            {   InterpolateMode::LINEAR_ONNX,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
            {2.6666665f, 4.3333331f},
        },
        {   "linear_onnx.resize_downsample_sizes_linear_pytorch_half_pixel",
            Shape{1, 1, 4, 4},
            {3, 1},
            Shape{1, 1, 3, 1},
            {0.75f, 0.25f},
            {2, 3},
            {   InterpolateMode::LINEAR_ONNX,
                ShapeCalcMode::SIZES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::PYTORCH_HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
            {1.6666666f, 7.0f, 12.333333f},
        },
        {   "linear_onnx.resize_upsample_scales_linear",
            Shape{1, 1, 2, 2},
            {4, 4},
            Shape{1, 1, 4, 4},
            {2.0f, 2.0f},
            {2, 3},
            {   InterpolateMode::LINEAR_ONNX,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.0f, 1.25f, 1.75f, 2.0f, 1.5f, 1.75f, 2.25f, 2.5f, 2.5f, 2.75f, 3.25f, 3.5f, 3.0f, 3.25f, 3.75f, 4.0f},
        },
        {   "linear_onnx.resize_upsample_scales_linear_align_corners",
            Shape{1, 1, 2, 2},
            {4, 4},
            Shape{1, 1, 4, 4},
            {2.0f, 2.0f},
            {2, 3},
            {   InterpolateMode::LINEAR_ONNX,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::ALIGN_CORNERS,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.0f, 1.33333333f, 1.66666667f, 2.0f, 1.66666667f, 2.0f, 2.33333333f, 2.66666667f,
             2.33333333f, 2.66666667f, 3.0f, 3.33333333f, 3.0f, 3.33333333f, 3.66666667f, 4.0f},
        },
        {   "linear_onnx.resize_downsample_scales_linear_align_corners",
            Shape{1, 1, 2, 4},
            {1, 2},
            Shape{1, 1, 1, 2},
            {0.6f, 0.6f},
            {2, 3},
            {   InterpolateMode::LINEAR_ONNX,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::ALIGN_CORNERS,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
            {1.0f, 4.0f},
        }
    };
    // clang-format on
}

std::vector<InterpolateV4TestParams> generateParamsForInterpolate_v4_linear_onnx5d() {
    const std::vector<size_t> zero_pads{0, 0, 0, 0, 0};
    // clang-format off
    return {
        {   "linear_onnx5d.resize_downsample_scales_linear",
            Shape{1, 1, 3, 2, 4},
            {2, 1, 2},
            Shape{1, 1, 2, 1, 2},
            {0.8f, 0.6f, 0.6f},
            {2, 3, 4},
            {   InterpolateMode::LINEAR_ONNX,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
             13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
            {3.6666665, 5.333333, 13.666666, 15.333333}
        },
        {   "linear_onnx5d.resize_downsample_scales_linear_align_corners",
            Shape{1, 1, 3, 2, 4},
            {2, 1, 2},
            Shape{1, 1, 2, 1, 2},
            {0.8f, 0.6f, 0.6f},
            {2, 3, 4},
            {   InterpolateMode::LINEAR_ONNX,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::ALIGN_CORNERS,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
             13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
            {1.0, 4.0, 17.0, 20.0}
        },
        {   "linear_onnx5d.resize_upsample_scales_linear",
            Shape{1, 1, 2, 2, 2},
            {4, 4, 4},
            Shape{1, 1, 4, 4, 4},
            {2.0, 2.0, 2.0},
            {2, 3, 4},
            {   InterpolateMode::LINEAR_ONNX,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
            {1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0,
             2.0, 2.25, 2.75, 3.0, 2.5, 2.75, 3.25, 3.5, 3.5, 3.75, 4.25, 4.5, 4.0, 4.25, 4.75, 5.0,
             4.0, 4.25, 4.75, 5.0, 4.5, 4.75, 5.25, 5.5, 5.5, 5.75, 6.25, 6.5, 6.0, 6.25, 6.75, 7.0,
             5.0, 5.25, 5.75, 6.0, 5.5, 5.75, 6.25, 6.5, 6.5, 6.75, 7.25, 7.5, 7.0, 7.25, 7.75, 8.0}
        },
        {   "linear_onnx5d.resize_upsample_scales_linear_align_corners",
            Shape{1, 1, 2, 2, 2},
            {4, 4, 4},
            Shape{1, 1, 4, 4, 4},
            {2.0, 2.0, 2.0},
            {2, 3, 4},
            {   InterpolateMode::LINEAR_ONNX,
                ShapeCalcMode::SCALES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::ALIGN_CORNERS,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
            {1.0,       1.3333333, 1.6666667, 2.0,       1.6666666, 2.0,       2.3333335, 2.6666667, 2.3333333, 2.6666665,
             3.0,       3.3333335, 3.0,       3.3333333, 3.6666665, 4.0,       2.3333335, 2.6666665, 3.0,       3.3333333,
             3.0,       3.333333,  3.6666665, 3.9999995, 3.6666665, 4.0,       4.3333335, 4.6666665, 4.333333,  4.6666665,
             4.9999995, 5.333333,  3.6666667, 4.0,       4.3333335, 4.6666665, 4.3333335, 4.6666665, 5.0,       5.333333,
             5.0,       5.3333335, 5.666667,  6.0,       5.666667,  5.9999995, 6.333333,  6.666667,  5.0,       5.333333,
             5.6666665, 6.0,       5.666667,  5.9999995, 6.333333,  6.666666,  6.3333335, 6.666666,  7.0,       7.3333335,
             7.0,       7.333333,  7.6666675, 8.0}
        },
        {   "linear_onnx5d.resize_downsample_sizes_linear_pytorch_half_pixel",
            Shape{1, 1, 2, 4, 4},
            {1, 3, 1},
            Shape{1, 1, 1, 3, 1},
            {0.5, 0.75, 0.25},
            {2, 3, 4},
            {   InterpolateMode::LINEAR_ONNX,
                ShapeCalcMode::SIZES,
                zero_pads,
                zero_pads,
                CoordinateTransformMode::PYTORCH_HALF_PIXEL,
                NearestMode::ROUND_PREFER_FLOOR},
            {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
             12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f,
             23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f},
            {1.6666667, 7.0, 12.333333}
        }
    };
    // clang-format on
}

std::vector<InterpolateV4TestParams> generateCombinedParamsForInterpolate_v4() {
    const std::vector<std::vector<InterpolateV4TestParams>> allTypeParams{
        generateParamsForInterpolate_v4_cubic(),
        generateParamsForInterpolate_v4_nearest(),
        generateParamsForInterpolate_v4_linear_onnx(),
        generateParamsForInterpolate_v4_linear_onnx5d()};

    std::vector<InterpolateV4TestParams> combinedParams;
    for (auto& params : allTypeParams)
        std::move(params.begin(), params.end(), std::back_inserter(combinedParams));
    return combinedParams;
}

class ReferenceInterpolate_v4 : public testing::TestWithParam<InterpolateV4TestParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& param = GetParam();
        function = CreateFunction(param);
        inputData = {CreateTensor(param.input_data_shape, element::f32, param.input_data)};
        refOutData = {CreateTensor(param.output_shape, element::f32, param.expected_results)};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateV4TestParams>& obj) {
        return obj.param.test_name;
    }

private:
    static std::shared_ptr<Model> CreateFunction(const InterpolateV4TestParams& param) {
        auto image = std::make_shared<op::v0::Parameter>(element::f32, param.input_data_shape);
        const auto& spatial_shape_data = param.spatial_shape_data;
        auto target_spatial_shape =
            op::v0::Constant::create<int64_t>(element::i64, Shape{spatial_shape_data.size()}, spatial_shape_data);
        const auto& scales_data = param.scales_data;
        auto scales = op::v0::Constant::create<float>(element::f32, Shape{scales_data.size()}, scales_data);
        const auto& axes_data = param.axes_data;
        auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{axes_data.size()}, axes_data);
        auto interpolate = std::make_shared<op::v4::Interpolate>(image, target_spatial_shape, scales, axes, param.attrs);
        return std::make_shared<Model>(NodeVector{interpolate}, ParameterVector{image});
    }
};

TEST_P(ReferenceInterpolate_v4, LayerTest) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ReferenceInterpolate_v4,
                         ::testing::ValuesIn(generateCombinedParamsForInterpolate_v4()),
                         ReferenceInterpolate_v4::getTestCaseName);

}  // namespace attribute_tests
}  // namespace
