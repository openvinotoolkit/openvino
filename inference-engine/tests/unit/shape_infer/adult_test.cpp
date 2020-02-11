// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <tuple>
#include "adult_test.hpp"
#include "debug.h"
#include <cmath>

using namespace InferenceEngine;
using namespace details;
using namespace ShapeInfer;
using namespace ShapeInferTests;

void BasicTest::SetUp() {
    auto params = GetParam();
    type = std::get<0>(params);
    inOutData = std::get<1>(params);
}

void BlobTest::SetUp() {
    auto params = GetParam();
    type = std::get<0>(params);
    inOutData = std::get<1>(params);
    blobsParam = std::get<2>(params);
}

void ParamsTest::SetUp() {
    auto params = GetParam();
    type = std::get<0>(params);
    inOutData = std::get<1>(params);
    strParams = std::get<2>(params);
}

ASITestBuilder CommonTests::assertThat() {
    return ASITestBuilder().withType(type).withData(inOutData);
}

std::vector<Precision> StridedSliceTest::getPrecisions() {
    size_t size = inOutData.inData.size();
    std::vector<Precision> result;
    if (!size) THROW_IE_EXCEPTION << "unsupported number of precisions";
    result.emplace_back(Precision::FP32);
    for (int i = 1; i < size; i++) {
        result.emplace_back(Precision::I32);
    }
    return result;
}

std::vector<float> FillTest::refGen(const InOutData& inOutData) {
    const size_t FILL_DIMS = 0;
    const size_t FILL_VALUE = 1;
    float value = inOutData.inData[FILL_VALUE][0];
    auto shape = inOutData.inData[FILL_DIMS];
    return std::vector<float>(product(shape), value);
}

std::vector<float> RangeTest::refGen(const InOutData& inOutData) {
    std::vector<float> result;
    float start = inOutData.inData[0][0];
    float limit = inOutData.inData[1][0];
    float delta = inOutData.inData[2][0];
    size_t work_amount_dst = std::floor(std::abs((limit - start) / delta));
    if (work_amount_dst != product(inOutData.inOutShapes.outDims[0]))
        THROW_IE_EXCEPTION << "Range indexes exceeds data tensor dimension";

    float dst_value = start;
    for (size_t iwork = 0; iwork < work_amount_dst; ++iwork, dst_value += delta) {
        result.push_back(dst_value);
    }
    return result;
}

std::vector<float> BroadcastTest::refGen(const InOutData& inOutData) {
    const size_t BROADCAST_DIMS = 0;
    const size_t BROADCAST_VALUE = 1;
    float value = inOutData.inData[BROADCAST_VALUE][0];
    auto shape = inOutData.inData[BROADCAST_DIMS];
    return std::vector<float>(product(shape), value);
}

TEST_P(BlobTest, impl) {
    assertThat().constInferResultFor().withBlobs(blobsParam).equals().toData(inOutData.outData);
}

TEST_P(BasicTest, impl) {
    assertThat().constInferResultFor().equals().toData(inOutData.outData);
}

TEST_P(ParamsTest, impl) {
    assertThat().constInferResultFor().withParams(strParams.data).equals().toData(inOutData.outData);
}

TEST_P(StridedSliceTest, impl) {
    assertThat().constInferResultFor().withParams(strParams.data)
            .withInputPrecisions(getPrecisions()).equals().toData(inOutData.outData);
}

TEST_P(StridedSliceTest, shapeInfer) {
    assertThat().shapeInferResultFor().withParams(strParams.data)
            .withInputPrecisions(getPrecisions())
            .equals().toShapes(inOutData.inOutShapes.outDims);
}

TEST_P(BasicAdultTest, impl) {
    assertThat().shapeInferResultFor().equals().toShapes(inOutData.inOutShapes.outDims);
}

TEST_P(FillTest, impl) {
    assertThat().constInferResultFor().withInputPrecisions({Precision::I32, Precision::FP32})
            .equals().toData({refGen(inOutData)});
}

TEST_P(FillTest, shapeInfer) {
    assertThat().shapeInferResultFor().withInputPrecisions({Precision::I32, Precision::FP32})
            .equals().toShapes(inOutData.inOutShapes.outDims);
}

TEST_P(RangeTest, impl) {
    assertThat().constInferResultFor().equals().toData({refGen(inOutData)});
}

TEST_P(RangeTest, shapeInfer) {
    assertThat().shapeInferResultFor().equals().toShapes(inOutData.inOutShapes.outDims);
}

TEST_P(BroadcastTest, impl) {
    assertThat().constInferResultFor().withInputPrecisions({Precision::FP32, Precision::I32})
            .equals().toData({refGen(inOutData)});
}

TEST_P(BroadcastTest, shapeInfer) {
    assertThat().shapeInferResultFor().withInputPrecisions({Precision::FP32, Precision::I32})
            .equals().toShapes(inOutData.inOutShapes.outDims);
}

static std::vector<float> singleInputData = {4.f, 8.f, 12.f, 16.f};

static testing::InOutShapes singleSmallShapes = {{{1, 3}},
                                                 {{1, 3}}};
static std::vector<float> singleSmallData = {1.f, 2.f, 4.f};

static testing::InOutShapes singleSmall2Shapes = {{{1, 3}, {1, 3}},
                                                  {{1, 3}}};

static testing::InOutShapes singleInOutShape = {{{4, 8, 12, 16}},
                                                {{4}}};

static std::vector<float> fourInARow = {1.f, 2.f, 3.f, 4.f};

static SizeVector threeDeuces = {2, 2, 2};

INSTANTIATE_TEST_CASE_P(
        CheckOutputDirectly, BlobTest,
        ::testing::Values(
                ::testing::make_tuple(LayerType("Const"), InOutDataParam({singleInOutShape, {}, {singleInputData}}),
                                      BlobsParam(FloatMap{{"custom", singleInputData}}))
        )
);

INSTANTIATE_TEST_CASE_P(
        CheckOutputDirectly, ParamsTest,
        ::testing::Values(
                ::testing::make_tuple(LayerType("Power"),
                                      InOutDataParam({singleSmallShapes,
                                                      {singleSmallData},
                                                      {{-2 / 3.f, -2 / 7.f, -2 / 15.f}}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"power", "-1"},
                                                                                             {"scale", "-2"},
                                                                                             {"shift", "0.5"}}))),
                ::testing::make_tuple(LayerType("Power"),
                                      InOutDataParam({singleSmallShapes,
                                                      {singleSmallData},
                                                      {{-3.375f, -1.f, 0.f,}}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"power", "3"},
                                                                                             {"scale", "0.5"},
                                                                                             {"shift", "-2"}}))),
                ::testing::make_tuple(LayerType("Power"),
                                      InOutDataParam({singleSmallShapes,
                                                      {singleSmallData},
                                                      {{10.f, 10.f, 10.f,}}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"power", "1"},
                                                                                             {"scale", "0"},
                                                                                             {"shift", "10"}}))),
                ::testing::make_tuple(LayerType("Tile"),
                                      InOutDataParam({{{{2, 1, 2}},
                                                              {threeDeuces}},
                                                      {fourInARow},
                                                      {{1.f, 2.f, 1.f, 2.f, 3.f, 4.f, 3.f, 4.f}}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis",  "1"},
                                                                                             {"tiles", "2"}}))),
                ::testing::make_tuple(LayerType("Tile"),
                                      InOutDataParam({{{{2, 2, 1}},
                                                              {threeDeuces}},
                                                      {fourInARow},
                                                      {{1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f}}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis",  "2"},
                                                                                             {"tiles", "2"}}))),
                ::testing::make_tuple(LayerType("Tile"),
                                      InOutDataParam({{{{1, 2, 2}},
                                                              {threeDeuces}},
                                                      {fourInARow},
                                                      {{1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f}}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis",  "0"},
                                                                                             {"tiles", "2"}}))),
                ::testing::make_tuple(LayerType("Reshape"),
                                      InOutDataParam({{{{1, 2, 2}}, {{4}}},
                                                      {fourInARow},
                                                      {fourInARow}}),
                                      MapParams(MapStrStr())),
                ::testing::make_tuple(LayerType("Split"),
                                      InOutDataParam({{{{2, 1, 2}}, {{2, 1, 1}, {2, 1, 1}}},
                                                      {fourInARow},
                                                      {{1.f, 3.f},  {2.f,       4.f}}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis", "2"}}))),
                ::testing::make_tuple(LayerType("Split"),
                                      InOutDataParam({{{{2, 1, 2}}, {{1, 1, 2}, {1, 1, 2}}},
                                                      {fourInARow},
                                                      {{1.f, 2.f},  {3.f,       4.f}}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis", "0"}}))),
                ::testing::make_tuple(LayerType("Split"),
                                      InOutDataParam({{{{4, 1, 1}}, {{2, 1, 1}, {1, 1, 1}, {1, 1, 1}}},
                                                      {fourInARow},
                                                      {{1.f, 2.f},  {3.f}, {4.f}}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis", "0"}}))),
                ::testing::make_tuple(LayerType("Concat"),
                                      InOutDataParam({{{{2, 1, 1}, {2, 1, 1}}, {{2, 1, 2}}},
                                                      {{1.f,       3.f},       {2.f, 4.f}},
                                                      {fourInARow}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis", "2"}}))),
                ::testing::make_tuple(LayerType("Concat"),
                                      InOutDataParam({{{{1, 1, 2}, {1, 1, 2}}, {{2, 1, 2}}},
                                                      {{1.f,       2.f},       {3.f, 4.f}},
                                                      {fourInARow}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis", "0"}}))),
                ::testing::make_tuple(LayerType("Concat"),
                                      InOutDataParam({{{{2, 1, 1}, {1, 1, 1}, {1, 1, 1}}, {{4, 1, 1}}},
                                                      {{1.f,       2.f},                  {3.f}, {4.f}},
                                                      {fourInARow}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis", "0"}})))
        )
);

namespace {
//  Test data vectors
std::vector<float> in0 = {0.f, 1.f, 1.f, 0.f};
std::vector<float> in1 = {0.f, 1.f, 2.f, 1.f};
std::vector<float> dict = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
std::vector<float> dict2D = {1.f, 2.f, 3.f, 4.f}; // 2x2
std::vector<float> ref_in0_a0_d223 = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 7.f, 8.f, 9.f,
                                      10.f, 11.f, 12.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f}; // 2x2x2x3
std::vector<float> ref_in1_a2_d223 = {1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f,
                                      11.f}; // 2x2x2x2
std::vector<float> ref_in0_a0_d22 = {1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f}; // 2x2x2
}

INSTANTIATE_TEST_CASE_P(
        TestsGather, ParamsTest,
        ::testing::Values(
                ::testing::make_tuple(LayerType("Gather"),
                                      InOutDataParam({{{{2, 2}, {1, 4}}, {{1, 4, 2}}},
                                                      {dict2D,           in0},
                                                      {ref_in0_a0_d22}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis", "0"}}))),
                ::testing::make_tuple(LayerType("Gather"),
                                      InOutDataParam({{{{2, 2, 3}, {2, 2}}, {{2, 2, 2, 3}}},
                                                      {dict,                in0},
                                                      {ref_in0_a0_d223}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis", "0"}}))),
                ::testing::make_tuple(LayerType("Gather"),
                                      InOutDataParam({{{{2, 2, 3}, {2, 2}}, {{2, 2, 2, 3}}},
                                                      {dict,                in0},
                                                      {ref_in0_a0_d223}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis", "-3"}}))),
                ::testing::make_tuple(LayerType("Gather"),
                                      InOutDataParam({{{{2, 2, 3}, {2, 2}}, {{2, 2, 2, 2}}},
                                                      {dict,                in1},
                                                      {ref_in1_a2_d223}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"axis", "2"}})))
        )
);

//static testing::InOutShapes eltWiseShapes1 = {{{4}, {1}},
//                                              {{4}}};
//static std::vector<std::vector<float>> eltWiseInputs1 = {singleInputData,
//                                                         {4.f}};
//
//static testing::InOutShapes eltWiseShapes2 = {{{2, 3}, {3}},
//                                              {{2, 3}}};
//static std::vector<std::vector<float>> eltWiseInputs2 = {{4.f, 8.f, 12.f, 4.f, 8.f, 8.f},
//                                                         {4.f, 8.f, 4.f}};
INSTANTIATE_TEST_CASE_P(
        CheckOutputDirectly, BasicTest,
        ::testing::Values(
                ::testing::make_tuple(
                        LayerType("Shape"),
                        InOutDataParam({singleInOutShape, {}, {singleInputData}})),
//                ::testing::make_tuple(
//                        LayerType("Mul"),
//                        InOutDataParam({eltWiseShapes1, eltWiseInputs1, {{16.f, 32.f, 48.f, 64.f}}})),
//                ::testing::make_tuple(
//                        LayerType("Add"),
//                        InOutDataParam({eltWiseShapes1, eltWiseInputs1, {{8.f, 12.f, 16.f, 20.f}}})),
//                ::testing::make_tuple(
//                        LayerType("Div"),
//                        InOutDataParam({eltWiseShapes1, eltWiseInputs1, {{1.f, 2.f, 3.f, 4.f}}})),
//                ::testing::make_tuple(
//                        LayerType("Mul"),
//                        InOutDataParam({eltWiseShapes2, eltWiseInputs2, {{16.f, 64.f, 48.f, 16.f, 64.f, 32.f}}})),
//                ::testing::make_tuple(
//                        LayerType("Add"),
//                        InOutDataParam({eltWiseShapes2, eltWiseInputs2, {{8.f, 16.f, 16.f, 8.f, 16.f, 12.f}}})),
//                ::testing::make_tuple(
//                        LayerType("Div"),
//                        InOutDataParam({eltWiseShapes2, eltWiseInputs2, {{1.f, 1.f, 3.f, 1.f, 1.f, 2.f}}})),
                ::testing::make_tuple(LayerType("Mul"),
                                      InOutDataParam({singleSmall2Shapes, {singleSmallData, singleSmallData},
                                                      {{1.f, 4.f, 16.f}}})),
                ::testing::make_tuple(LayerType("Add"),
                                      InOutDataParam({singleSmall2Shapes, {singleSmallData, singleSmallData},
                                                      {{2.f, 4.f, 8.f}}})),
                ::testing::make_tuple(LayerType("Div"),
                                      InOutDataParam({singleSmall2Shapes, {singleSmallData, singleSmallData},
                                                      {{1.f, 1.f, 1.f}}}))
        )
);

INSTANTIATE_TEST_CASE_P(
        SecondInput, BasicAdultTest,
        ::testing::Combine(::testing::Values(LayerType("Reshape"), LayerType("Interp"), LayerType("Resample")),
                           ::testing::Values(InOutDataParam({{{{2, 3}, {2}},
                                                                     {{1, 6}}},
                                                             {{},    {1.f, 6.f}},
                                                             {}})))
);

INSTANTIATE_TEST_CASE_P(
        DimSemantic, BasicAdultTest,
        ::testing::Values(
                ::testing::make_tuple(LayerType("Reshape"),
                                      InOutDataParam({{{{2, 3}, {2}},
                                                              {{1, 6}}},
                                                      {{},    {1.f, -1.f}},
                                                      {}}))
        )
);

INSTANTIATE_TEST_CASE_P(
        SqueezeUnsqueeze, BasicAdultTest,
        ::testing::Values(
                ::testing::make_tuple(LayerType("Unsqueeze"),
                                      InOutDataParam({{{{3}, {1}},
                                                              {{1, 3}}},
                                                      {{},    {0.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Unsqueeze"),
                                      InOutDataParam({{{{3}, {3}},
                                                              {{1, 1, 1, 3}}},
                                                      {{},    {0.f, 1.f, 2.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Unsqueeze"),
                                      InOutDataParam({{{{3}, {3}},
                                                              {{1, 3, 1, 1}}},
                                                      {{},    {0.f, 2.f, 3.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Unsqueeze"),
                                      InOutDataParam({{{{2, 3}, {2}},
                                                              {{1, 2, 3, 1}}},
                                                      {{},    {0.f, 3.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Unsqueeze"),
                                      InOutDataParam({{{{2, 3}, {1}},
                                                              {{2, 1, 3}}},
                                                      {{},    {1.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Unsqueeze"),
                                      InOutDataParam({{{{3}, {1}},
                                                              {{1, 3}}},
                                                      {{},    {0.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Unsqueeze"),
                                      InOutDataParam({{{{3}, {3}},
                                                              {{1, 1, 1, 3}}},
                                                      {{},    {0.f, 1.f, 2.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Unsqueeze"),
                                      InOutDataParam({{{{3}, {3}},
                                                              {{1, 3, 1, 1}}},
                                                      {{},    {0.f, 2.f, 3.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Unsqueeze"),
                                      InOutDataParam({{{{2, 3}, {2}},
                                                              {{1, 2, 3, 1}}},
                                                      {{},    {0.f, 3.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Unsqueeze"),
                                      InOutDataParam({{{{2, 3}, {1}},
                                                              {{2, 1, 3}}},
                                                      {{},    {1.f,}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Squeeze"),
                                      InOutDataParam({{{{1}, {1}},
                                                              {{}}},
                                                      {{},    {0.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Squeeze"),
                                      InOutDataParam({{{{1, 3, 1}, {1}},
                                                              {{3, 1}}},
                                                      {{},    {0.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Squeeze"),
                                      InOutDataParam({{{{1, 3, 1}, {1}},
                                                              {{1, 3}}},
                                                      {{},    {2.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Squeeze"),
                                      InOutDataParam({{{{1, 3, 1}, {2}},
                                                              {{3}}},
                                                      {{},    {0.f, 2.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Squeeze"),
                                      InOutDataParam({{{{1, 3, 1}, {1}},
                                                              {{1, 3}}},
                                                      {{},    {-1.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Squeeze"),
                                      InOutDataParam({{{{1, 3, 1, 2}, {2}},
                                                              {{3, 2}}},
                                                      {{},    {0.f, 2.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Squeeze"),
                                      InOutDataParam({{{{1}, {1}},
                                                              {{}}},
                                                      {{},    {0.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Squeeze"),
                                      InOutDataParam({{{{1, 3, 1}, {1}},
                                                              {{1, 3}}},
                                                      {{},    {2.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Squeeze"),
                                      InOutDataParam({{{{1, 3, 1}, {2}},
                                                              {{3}}},
                                                      {{},    {0.f, 2.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Squeeze"),
                                      InOutDataParam({{{{1, 3, 1}, {1}},
                                                              {{1, 3}}},
                                                      {{},    {-1.f}},
                                                      {}})),
                ::testing::make_tuple(LayerType("Squeeze"),
                                      InOutDataParam({{{{1, 3, 1, 2}, {2}},
                                                              {{3, 2}}},
                                                      {{},    {0.f, 2.f}},
                                                      {}}))
        )
);
namespace {
//  Test data vectors
std::vector<float> test0 = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
std::vector<float> test2 = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
std::vector<float> test5 = {5.f, 6.f, 7.f, 8.f};
std::vector<float> test6 = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
std::vector<float> test8 = {5.f, 4.f, 3.f, 2.f, 1.f};
std::vector<float> test9 = {5.f, 4.f, 3.f, 2.f, 1.f, 0.f};
std::vector<float> test10 = {5.f, 4.f, 3.f};
std::vector<float> test11 = {0.f, 2.f, 4.f, 6.f, 8.f};
std::vector<float> test12 = {1.f, 3.f, 5.f, 7.f, 9.f};
std::vector<float> test13 = {9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f};
std::vector<float> test14 = {9.f, 7.f, 5.f, 3.f, 1.f};
std::vector<float> test16 = {0.f, 1.f, 3.f, 4.f};
std::vector<float> test17 = {1.f, 4.f};
std::vector<float> test19 = {0.f, 1.f, 2.f, 3.f};
std::vector<float> test20 = {4.f, 5.f, 6.f, 7.f};
/*
0. [0,1,2,3,4,5,6,7,8,9], shape=[10]
1. [0,1,2,3,4,5,6,7,8,9], shape=[10]
2. [0,1,2,3,4,5,6,7,8], shape=[9]
3. [0,1,2,3,4,5,6,7,8], shape=[9]
4. [0,1,2,3,4,5,6,7,8,9], shape=[10]
5. [5,6,7,8,9], shape=[5]
6. [0,1,2,3,4,5], shape=[6]
7. [5,6,7,8,9], shape=[5]
8. [5,4,3,2,1], shape=[5]
9. [5,4,3,2,1,0], shape=[6]
10. [5,4,3], shape=[3]
11. [0,2,4,6,8], shape=[5]
12. [1,3,5,7,9], shape=[5]
13. [9,8,7,6,5,4,3,2,1,0], shape=[10]
14. [9,7,5,3,1], shape=[5]
15. [[0,1,2,3,4,5,6,7,8,9]], shape=[1,10]
16. [[[0,1,2],[3,4,5]]], shape=[1,2,2]
17. [[[0,1,2],[3,4,5]]], shape=[1,2,1]
18. [[[0,1,2],[3,4,5]]], shape=[1,1,2,1]
19. [[[[0,1],[2,3]],[[4,5],[6,7]]]], shape=[1,2,2]
20. [[[[0,1],[2,3]],[[4,5],[6,7]]]], shape=[1,2,2]
21. [[[0,1,2],[3,4,5]]], shape=[1,1,2]
*/
}

INSTANTIATE_TEST_CASE_P(
        StridedSlice, StridedSliceTest,
        ::testing::Values(
                /* 0 */
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {}, {}, {}}, {{10}}},
                                                                                 {{test0},            {}, {}, {}},
                                                                                 {test0}}),
                                      MapParams(MapStrStr())),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {}}, {{10}}},
                                                                                 {{test0},              {0.f}, {0.f}, {}},
                                                                                 {test0}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"end_mask", "0"}}))),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {}}, {{9}}},
                                                                                 {{test0},              {-1.f}, {-1.f}, {}},
                                                                                 {test2}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"begin_mask", "0"}}))),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {}}, {{9}}},
                                                                                 {{test0},              {0.f}, {-1.f}, {}},
                                                                                 {test2}}),
                                      MapParams(MapStrStr())),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {}}, {{10}}},
                                                                                 {{test0},              {0.f}, {10.f}, {}},
                                                                                 {test0}}),
                                      MapParams(MapStrStr())),
/* 5 */
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {}}, {{5}}},
                                                                                 {{test0},              {5.f}, {10.f}, {}},
                                                                                 {test5}}),
                                      MapParams(MapStrStr())),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {}}, {{6}}},
                                                                                 {{test0},              {0.f}, {6.f}, {}},
                                                                                 {test6}}),
                                      MapParams(MapStrStr())),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {}}, {{5}}},
                                                                                 {{test0},              {-5.f}, {10.f}, {}},
                                                                                 {test5}}),
                                      MapParams(MapStrStr())),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {1}}, {{5}}},
                                                                                 {{test0},               {-5.f}, {0.f}, {-1.f}},
                                                                                 {test8}}),
                                      MapParams(MapStrStr())),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {1}}, {{6}}},
                                                                                 {{test0},               {-5.f}, {0.f}, {-1.f}},
                                                                                 {test9}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"end_mask", "0"}}))
                ),
/* 10 */
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {1}}, {{3}}},
                                                                                 {{test0},               {-5.f}, {2.f}, {-1.f}},
                                                                                 {test10}}),
                                      MapParams(MapStrStr())),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {1}}, {{5}}},
                                                                                 {{test0},               {0.f}, {0.f}, {2.f}},
                                                                                 {test11}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"end_mask", "0"}}))),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {1}}, {{5}}},
                                                                                 {{test0},               {1.f}, {0.f}, {2.f}},
                                                                                 {test12}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"end_mask", "0"}}))),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {1}}, {{10}}},
                                                                                 {{test0},               {-1.f}, {0.f}, {-1.f}},
                                                                                 {test13}}),
                                      MapParams(MapStrStr(
                                              std::map<std::string, std::string>{{"end_mask", "0"}}))),
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {1}}, {{5}}},
                                                                                 {{test0},               {-1.f}, {0.f}, {-2.f}},
                                                                                 {test14}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"end_mask", "0"}}))),
/* 15 */
                ::testing::make_tuple(LayerType("StridedSlice"), InOutDataParam({{{{10}, {1}, {1}, {}}, {{1, 10}}},
                                                                                 {{test0},              {0.f}, {10.f}, {}},
                                                                                 {test0}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"new_axis_mask", "1"}}))),
                ::testing::make_tuple(LayerType("StridedSlice"),
                                      InOutDataParam({{{{1, 2, 3}, {2}, {2}, {}}, {{1, 2, 2}}},
                                                      {{test0},                   {0.f, 0.f}, {1.f, 2.f}, {}},
                                                      {test16}}),
                                      MapParams(
                                              MapStrStr(std::map<std::string, std::string>{{"ellipsis_mask", "0,1"}}))),
                ::testing::make_tuple(LayerType("StridedSlice"),
                                      InOutDataParam({{{{1, 2, 3}, {4}, {4}, {}}, {{1,   2,   1}}},
                                                      {{test0},                   {{0.f, 0.f, 0.f, 1.f}}, {2.f, 3.f, 2.f, 2.f}, {}},
                                                      {test17}}),
                                      MapParams(
                                              MapStrStr(std::map<std::string, std::string>{{"new_axis_mask",    "0,0,1,0"},
                                                                                           {"shrink_axis_mask", "0,0,0,1"}}))),
                ::testing::make_tuple(LayerType("StridedSlice"),
                                      InOutDataParam({{{{1, 2, 3}, {3}, {3}, {}}, {{1, 1, 2, 1}}},
                                                      {{test0},                   {0.f, 0.f, 1.f}, {2.f, 2.f, 2.f}, {}},
                                                      {test17}}),
                                      MapParams(MapStrStr(
                                              std::map<std::string, std::string>{{"ellipsis_mask", "0,1"},
                                                                                 {"new_axis_mask", "1"}}))),
                ::testing::make_tuple(LayerType("StridedSlice"),
                                      InOutDataParam({{{{1, 2, 2, 2}, {1}, {1}, {1}}, {{1, 2, 2}}},
                                                      {{test0},                       {-1.f}, {0.f}, {-2.f}},
                                                      {test19}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"begin_mask",       "0,1,0,0"},
                                                                                             {"end_mask",         "0,1,0,0"},
                                                                                             {"shrink_axis_mask", "0,1"}}))),
/* 20 */
                ::testing::make_tuple(LayerType("StridedSlice"),
                                      InOutDataParam({{{{1, 2, 2, 2}, {4}, {4}, {}}, {{1, 2, 2}}},
                                                      {{test0},                      {0.f, 1.f, 0.f, 0.f}, {1.f, 2.f, 2.f, 2.f}, {}},
                                                      {test20}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"begin_mask",       "0,1,0,0"},
                                                                                             {"end_mask",         "0,1,0,0"},
                                                                                             {"shrink_axis_mask", "0,1,0,0"}}))),
                ::testing::make_tuple(LayerType("StridedSlice"),
                                      InOutDataParam({{{{1, 2, 3}, {3}, {3}, {}}, {{1, 1, 2}}},
                                                      {{test0},                   {0.f, 0.f, 1.f}, {2.f, 2.f, 2.f}, {}},
                                                      {test17}}),
                                      MapParams(MapStrStr(std::map<std::string, std::string>{{"ellipsis_mask",    "0,1"},
                                                                                             {"new_axis_mask",    "1"},
                                                                                             {"shrink_axis_mask", "0,0,1"}})))
        )
);

INSTANTIATE_TEST_CASE_P(
        Fill, FillTest,
        ::testing::Values(
                ::testing::make_tuple(LayerType("Fill"), InOutDataParam({{{{1}, {1}},
                                                                                 {{1}}},
                                                                         {{1.f}, {1.f}},
                                                                         {}})),
                ::testing::make_tuple(LayerType("Fill"), InOutDataParam({{{{3}, {1}},
                                                                                 {{1, 3, 1}}},
                                                                         {{1.f, 3.f, 1.f}, {1.f}},
                                                                         {}})),
                ::testing::make_tuple(LayerType("Fill"), InOutDataParam({{{{3}, {1}},
                                                                                 {{2, 3, 6}}},
                                                                         {{2.f, 3.f, 6.f}, {-1.f}},
                                                                         {}})),
                ::testing::make_tuple(LayerType("Fill"), InOutDataParam({{{{4}, {1}},
                                                                                 {{1, 3, 1, 2}}},
                                                                         {{1.f, 3.f, 1.f, 2.f}, {.5f}},
                                                                         {}})),
                ::testing::make_tuple(LayerType("Fill"), InOutDataParam({{{{6}, {1}},
                                                                                 {{4, 3, 2, 5, 4, 2}}},
                                                                         {{4.f, 3.f, 2.f, 5.f, 4.f, 2.f}, {.25f}},
                                                                         {}}))
        )
);

INSTANTIATE_TEST_CASE_P(
        Range, RangeTest,
        ::testing::Values(
                ::testing::make_tuple(LayerType("Range"), InOutDataParam({{{{1}, {1}, {1}},
                                                                                  {{5}}},
                                                                          {{3.f}, {18.f}, {3.f}},
                                                                          {{}}})),
                ::testing::make_tuple(LayerType("Range"), InOutDataParam({{{{1}, {1}, {1}},
                                                                                  {{2}}},
                                                                          {{3.f}, {1.f}, {-1.f}},
                                                                          {{}}})),
                ::testing::make_tuple(LayerType("Range"), InOutDataParam({{{{1}, {1}, {1}},
                                                                                  {{6}}},
                                                                          {{3.f}, {-3.f}, {-1.f}},
                                                                          {{}}})),
                ::testing::make_tuple(LayerType("Range"), InOutDataParam({{{{1}, {1}, {1}},
                                                                                  {{5}}},
                                                                          {{0.f}, {5.f}, {1.f}},
                                                                          {{}}}))
        )
);

INSTANTIATE_TEST_CASE_P(
        Broadcast, BroadcastTest,
        ::testing::Values(
                ::testing::make_tuple(LayerType("Broadcast"), InOutDataParam({{{{3}, {2}},
                                                                      {{3, 3}}},
                                                              {{},    {3, 3}},
                                                              {}})),
                ::testing::make_tuple(LayerType("Broadcast"), InOutDataParam({{{{16, 50, 1}, {4}},
                                                                      {{1, 16, 50, 50}}},
                                                              {{},    {1, 16, 50, 50}},
                                                              {}})),
                ::testing::make_tuple(LayerType("Broadcast"), InOutDataParam({{{{1}, {3}},
                                                                      {{1, 50, 50}}},
                                                              {{},    {1, 50, 50}},
                                                              {}}))
)
);