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

std::vector<InterpolateV4TestParams> generateParamsForInterpolate_bilinear_pil() {
    const std::vector<size_t> zero_pads{0, 0, 0, 0};
    return {
        {
            "bilinear.new_resize_downsample_scales_linear_range_h_pixel_hw_2D_sizes",
            Shape{8, 8},
            {4, 4},
            Shape{4, 4},
            {0.5f, 0.5f},
            {0, 1},
            {InterpolateMode::BILINEAR_PILLOW, ShapeCalcMode::SIZES, {0, 0}, {0, 0}},
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
             22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
            {7, 9, 11, 12, 21, 23, 25, 26, 37, 39, 41, 42, 51, 53, 55, 56},
        },
        {
            "bilinear.new_resize_downsample_scales_linear_range_h_pixel_hw_2D_scales",
            Shape{8, 8},
            {4, 4},
            Shape{4, 4},
            {0.5f, 0.5f},
            {0, 1},
            {InterpolateMode::BILINEAR_PILLOW, ShapeCalcMode::SCALES, {0, 0}, {0, 0}},
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
             22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
            {7, 9, 11, 12, 21, 23, 25, 26, 37, 39, 41, 42, 51, 53, 55, 56},
        },
        {
            "bilinear.new_resize_downsample_scales_linear_rand_h_pixel_nhwc",
            Shape{1, 4, 4, 3},
            {2, 2},
            Shape{1, 2, 2, 3},
            {0.5f, 0.5f},
            {1, 2},
            {InterpolateMode::BILINEAR_PILLOW, ShapeCalcMode::SCALES, zero_pads, zero_pads},
            {172, 10,  127, 140, 47,  170, 196, 151, 117, 166, 22,  183, 192, 204, 33,  216,
             67,  179, 78,  154, 251, 82,  162, 219, 195, 118, 125, 139, 103, 125, 229, 216,
             9,   164, 116, 108, 211, 222, 161, 159, 21,  81,  89,  165, 242, 214, 102, 98},
            {174, 97, 132, 144, 119, 173, 175, 129, 124, 160, 138, 129},
        },
        {
            "bilinear.new_resize_downsample_scales_linear_range_h_pixel_nhwc",
            Shape{1, 4, 4, 3},
            {2, 2},
            Shape{1, 2, 2, 3},
            {0.5f, 0.5f},
            {1, 2},
            {InterpolateMode::BILINEAR_PILLOW, ShapeCalcMode::SCALES, zero_pads, zero_pads},
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
             24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47},
            {11, 12, 13, 16, 17, 18, 29, 30, 31, 34, 35, 36},
        },
        {
            "bilinear.new_resize_downsample_scales_linear_rand_h_pixel_nhwc_batch_2",
            Shape{2, 4, 4, 3},
            {2, 2},
            Shape{2, 2, 2, 3},
            {0.5f, 0.5f},
            {1, 2},
            {InterpolateMode::BILINEAR_PILLOW, ShapeCalcMode::SCALES, zero_pads, zero_pads},
            {172, 10, 127, 140, 47,  170, 196, 151, 117, 166, 22,  183, 192, 204, 33,  216, 67,  179, 78,  154,
             251, 82, 162, 219, 195, 118, 125, 139, 103, 125, 229, 216, 9,   164, 116, 108, 211, 222, 161, 159,
             21,  81, 89,  165, 242, 214, 102, 98,  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,
             12,  13, 14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
             32,  33, 34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47},
            {174, 97, 132, 144, 119, 173, 175, 129, 124, 160, 138, 129, 11, 12, 13, 16, 17, 18, 29, 30, 31, 34, 35, 36},
        },
        {
            "bilinear.new_resize_downsample_sizes_nhwc_1x5x6x3_to_1x2x4x3",
            Shape{1, 5, 6, 3},
            {2, 4},
            Shape{1, 2, 4, 3},
            {0.5f, 0.5f},  // ignored
            {1, 2},
            {InterpolateMode::BILINEAR_PILLOW, ShapeCalcMode::SIZES, zero_pads, zero_pads},
            {37,  244, 193, 106, 235, 128, 71,  255, 140, 47,  103, 184, 72,  20,  188, 238, 255, 126,
             7,   0,   137, 195, 204, 32,  203, 170, 101, 77,  133, 30,  193, 255, 79,  203, 145, 37,
             192, 83,  112, 60,  144, 128, 163, 23,  129, 80,  134, 101, 204, 191, 174, 47,  71,  30,
             78,  99,  237, 170, 118, 88,  252, 121, 116, 171, 134, 141, 146, 101, 25,  125, 127, 239,
             178, 228, 239, 137, 20,  213, 167, 216, 254, 84,  80,  107, 101, 177, 50,  80,  146, 139},
            {89 /* 90 */, 137, 129, 138, 169, 107, 109, 140, 113, 168, 161, 95,
             134,         119, 178, 171, 118, 148, 138, 130, 106, 116, 133, 120},
        },
        {
            "bilinear.new_resize_upsample_sizes_nhwc_1x2x4x3_to_1x5x6x3",
            Shape{1, 2, 4, 3},
            {5, 6},
            Shape{1, 5, 6, 3},
            {0.5f, 0.5f},  // ignored
            {1, 2},
            {InterpolateMode::BILINEAR_PILLOW, ShapeCalcMode::SIZES, zero_pads, zero_pads},
            {37, 244, 193, 106, 235, 128, 71, 255, 140, 47,  103, 184,
             72, 20,  188, 238, 255, 126, 7,  0,   137, 195, 204, 32},
            {37,  244, 193,         72,  240, 161, 100, 238, 130, 77,  252, 138, 59,  179, 162, 47,
             103, 184, 41 /* 40 */, 222, 193, 80,  230, 161, 110, 235,  // Rounding?
             130, 74,  231,         138, 63,  171, 154, 62,  113, 169, 55,  132, 191, 114, 189, 159,
             150, 225, 129,         62,  148, 137, 80,  141, 124, 121, 154, 108, 69,  42,  188, 147,
             148, 157, 189,         215, 128, 49,  64,  135, 97,  110, 93,  180, 194, 47,  72,  20,
             188, 155, 138,         157, 199, 212, 128, 46,  43,  135, 101, 102, 85,  195, 204, 32},
        },
        {
            "bilinear.new_resize_downsample_sizes_nchw_1x3x5x6_to_1x3x2x4",
            Shape{1, 3, 5, 6},
            {2, 4},
            Shape{1, 3, 2, 4},
            {0.5f, 0.5f},  // ignored
            {2, 3},
            {InterpolateMode::BILINEAR_PILLOW, ShapeCalcMode::SIZES, zero_pads, zero_pads},
            {37,  106, 71,  47,  72,  238, 7,   195, 203, 77,  193, 203, 192, 60,  163, 80,  204, 47,
             78,  170, 252, 171, 146, 125, 178, 137, 167, 84,  101, 80,  244, 235, 255, 103, 20,  255,
             0,   204, 170, 133, 255, 145, 83,  144, 23,  134, 191, 71,  99,  118, 121, 134, 101, 127,
             228, 20,  216, 80,  177, 146, 193, 128, 140, 184, 188, 126, 137, 32,  101, 30,  79,  37,
             112, 128, 129, 101, 174, 30,  237, 88,  116, 141, 25,  239, 239, 213, 254, 107, 50,  139},
            {89 /* 90 */, 138, 109, 168, 134, 171, 138, 116, 137, 169, 140, 161,
             119,         118, 130, 133, 129, 107, 113, 95,  178, 148, 106, 120},
        },
        {
            "bilinear.new_resize_downsample_scales_range_h_pixel_nchw",
            Shape{1, 3, 4, 4},
            {2, 2},
            Shape{1, 3, 2, 2},
            {0.5f, 0.5f},
            {2, 3},
            {InterpolateMode::BILINEAR_PILLOW, ShapeCalcMode::SCALES, zero_pads, zero_pads},
            {0,  3,  6,  9,  12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 1,  4,  7,  10, 13, 16, 19, 22,
             25, 28, 31, 34, 37, 40, 43, 46, 2,  5,  8,  11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47},
            {11, 16, 29, 34, 12, 17, 30, 35, 13, 18, 31, 36},
        }};
}

std::vector<InterpolateV4TestParams> generateParamsForInterpolate_bicubic_pil() {
    const auto input_data_shape = Shape{1, 1, 4, 4};
    const std::vector<float> input_data =
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    const std::vector<size_t> zero_pads{0, 0, 0, 0};
    return {{
                "bicubic.new_resize_downsample_scales_2D",
                Shape{8, 8},
                {4, 4},
                Shape{4, 4},
                {0.5f, 0.5f},
                {0, 1},
                {InterpolateMode::BICUBIC_PILLOW, ShapeCalcMode::SCALES, {0, 0}, {0, 0}},
                {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
                {5, 6, 9, 10, 21, 22, 25, 26, 37, 38, 41, 42, 53, 54, 57, 58},
            },
            {
                "bicubic.new_resize_downsample_sizes_2D",
                Shape{8, 8},
                {4, 4},
                Shape{4, 4},
                {0.5f, 0.5f},
                {0, 1},
                {InterpolateMode::BICUBIC_PILLOW, ShapeCalcMode::SIZES, {0, 0}, {0, 0}},
                {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
                {5, 6, 9, 10, 21, 22, 25, 26, 37, 38, 41, 42, 53, 54, 57, 58},
            },
            {
                "bicubic.new_resize_downsample_sizes_1x1x8x8_nchw",
                Shape{1, 1, 8, 8},
                {4, 4},
                Shape{4, 4},
                {0.5f, 0.5f},
                {2, 3},
                {InterpolateMode::BICUBIC_PILLOW, ShapeCalcMode::SIZES, {0, 0}, {0, 0}},
                {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
                {5, 6, 9, 10, 21, 22, 25, 26, 37, 38, 41, 42, 53, 54, 57, 58},
            },
            {
                "bicubic.new_resize_downsample_sizes_1x8x8x1_nhwc",
                Shape{1, 8, 8, 1},
                {4, 4},
                Shape{4, 4},
                {0.5f, 0.5f},
                {1, 2},
                {InterpolateMode::BICUBIC_PILLOW, ShapeCalcMode::SIZES, {0, 0}, {0, 0}},
                {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
                {5, 6, 9, 10, 21, 22, 25, 26, 37, 38, 41, 42, 53, 54, 57, 58},
            },
            {
                "bicubic.new_resize_upsample_sizes_1x2x4x3_to_1x5x6x3_nhwc",
                Shape{1, 2, 4, 3},
                {5, 6},
                Shape{1, 5, 6, 3},
                {0.5f, 0.5f},  // ignored
                {1, 2},
                {InterpolateMode::BICUBIC_PILLOW, ShapeCalcMode::SIZES, {0, 0}, {0, 0}},
                {168, 92,  157, 111, 15, 138, 97,  47,  237, 25,  163, 6,
                 72,  118, 121, 238, 22, 174, 182, 140, 43,  121, 158, 242},
                {183, 94,  162, 141,         53,  141, 94,  11,  150, 93,       27,  255, 49,  105,
                 119, 10,  172, -35 /* 0 */, 165, 99,  155, 143, 55,  143,      116, 14,  // Clipped values
                 152, 108, 42,  226,         64,  113, 122, 26,  170, 8 /*17*/, 117, 111, 138, 148,
                 60,  148, 175, 22,          155, 148, 80,  143, 102, 133,      131, 69,  165, 123 /* 128 */,
                 68,  122, 121, 152,         65,  153, 233, 29,  158, 188,      118, 60,  140, 153,
                 140, 111, 160, 238,         50,  127, 114, 154, 67,  155,      255, 32,  160, 203,
                 133, 29,  155, 161,         143, 127, 158, 281 /* 255 */},
            },
            {
                "bicubic.new_resize_downsample_sizes_1x5x6x3_to_1x2x4x3_nhwc",
                Shape{1, 5, 6, 3},
                {2, 4},
                Shape{1, 2, 4, 3},
                {0.5f, 0.5f},  // ignored
                {1, 2},
                {InterpolateMode::BICUBIC_PILLOW, ShapeCalcMode::SIZES, {0, 0}, {0, 0}},
                {168, 92,  157, 111, 15,  138, 97,  47,  237, 25,  163, 6,   72,  118, 121, 238, 22,  174,
                 182, 140, 43,  121, 158, 242, 210, 73,  113, 111, 75,  132, 24,  124, 104, 57,  157, 107,
                 7,   173, 14,  82,  162, 210, 144, 84,  177, 129, 136, 39,  95,  218, 99,  52,  75,  170,
                 232, 178, 213, 138, 136, 158, 47,  20,  181, 30,  63,  43,  182, 76,  31,  125, 52,  124,
                 218, 202, 78,  68,  148, 25,  251, 161, 124, 160, 2,   159, 116, 78,  119, 209, 37,  219},
                {126, 125, 124, 133, 79,  181, 77,  127, 79, 95,  111, 131,
                 147, 178, 119, 124, 102, 144, 117, 75,  84, 135, 78,  134},
            }};
}

std::vector<InterpolateV4TestParams> generateCombinedParamsForInterpolate_v4() {
    const std::vector<std::vector<InterpolateV4TestParams>> allTypeParams{
        generateParamsForInterpolate_bilinear_pil(),
        generateParamsForInterpolate_bicubic_pil(),
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
        auto interpolate =
            std::make_shared<op::v4::Interpolate>(image, target_spatial_shape, scales, axes, param.attrs);
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
