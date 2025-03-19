// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/gather.hpp"

namespace {
using ov::test::Gather7LayerTest;
using ov::test::Gather8LayerTest;
using ov::test::Gather8withIndicesDataLayerTest;
using ov::test::GatherStringWithIndicesDataLayerTest;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::bf16,
        ov::element::i8
};

// Just need to check types transformation.
const std::vector<ov::element::Type> model_types_tr_check = {
        ov::element::i64,
        ov::element::f16
};

const std::vector<ov::Shape> input_shapes_1d = {{4}};

const std::vector<ov::Shape> indices_shapes_1d = {{1}, {3}};

const std::vector<std::tuple<int, int>> axes_batchdims_1d = {{0, 0}};

const auto gather7Params_1D = testing::Combine(
        testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_1d)),
        testing::ValuesIn(indices_shapes_1d),
        testing::ValuesIn(axes_batchdims_1d),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_1D, Gather7LayerTest, gather7Params_1D, Gather7LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TypesTrf, Gather7LayerTest,
            testing::Combine(
                testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_1d)),
                testing::ValuesIn(indices_shapes_1d),
                testing::ValuesIn(axes_batchdims_1d),
                testing::ValuesIn(model_types_tr_check),
                testing::Values(ov::test::utils::DEVICE_CPU)),
        Gather7LayerTest::getTestCaseName);

const std::vector<ov::Shape> input_shapes_2d = {{4, 19}};

const std::vector<ov::Shape> indices_shapes_2d = {{4, 1}, {4, 2}};

const std::vector<std::tuple<int, int>> axes_batchdims_2d = {{0, 0}, {1, 0}, {1, 1}, {-1, -1}};

const auto gather7Params_2D = testing::Combine(
        testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_2d)),
        testing::ValuesIn(indices_shapes_2d),
        testing::ValuesIn(axes_batchdims_2d),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_2D, Gather7LayerTest, gather7Params_2D, Gather7LayerTest::getTestCaseName);

const std::vector<ov::Shape> input_shapes_4d = {{4, 5, 6, 7}};

const std::vector<ov::Shape> indices_shapes_bd0 = {{4}, {2, 2}, {3, 3}, {5, 2}, {3, 2, 4}};

const std::vector<std::tuple<int, int>> axes_bd0 = {{0, 0}, {1, 0}, {2, 0}, {-1, 0}};

const auto gather7ParamsSubset_bd0 = testing::Combine(
        testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_4d)),
        testing::ValuesIn(indices_shapes_bd0),
        testing::ValuesIn(axes_bd0),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_bd0, Gather7LayerTest, gather7ParamsSubset_bd0, Gather7LayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_bd0, Gather8LayerTest, gather7ParamsSubset_bd0, Gather8LayerTest::getTestCaseName);

const std::vector<ov::Shape> indices_shapes_bd1 = {{4, 2}, {4, 5, 3}, {4, 1, 2, 3}};

const std::vector<std::tuple<int, int>> axes_bd1 = {{1, 1}, {2, 1}, {-1, 1}, {-2, 1}};

const auto gather7ParamsSubset_bd1 = testing::Combine(
        testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_4d)),
        testing::ValuesIn(indices_shapes_bd1),
        testing::ValuesIn(axes_bd1),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_bd1, Gather7LayerTest, gather7ParamsSubset_bd1, Gather7LayerTest::getTestCaseName);

const std::vector<ov::Shape> indices_shapes_bd2 = {{4, 5, 4, 3}, {4, 5, 3, 2}};

const std::vector<std::tuple<int, int>> axes_bd2 = {{2, 2}, {3, -2}, {-1, 2}, {-1, -2}};

const auto gather7ParamsSubset_bd2 = testing::Combine(
        testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_4d)),
        testing::ValuesIn(indices_shapes_bd2),
        testing::ValuesIn(axes_bd2),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_bd2, Gather7LayerTest, gather7ParamsSubset_bd2, Gather7LayerTest::getTestCaseName);

const std::vector<ov::Shape> indices_shapes_negative_bd = {{4, 5, 4}, {4, 5, 3}};

const std::vector<std::tuple<int, int>> axes_negative_bd = {{0, -3}, {1, -2}, {2, -2}, {-2, -2}, {-1, -1}, {-2, -1}};

const auto gather7ParamsSubset_negative_bd = testing::Combine(
        testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_4d)),
        testing::ValuesIn(indices_shapes_negative_bd),
        testing::ValuesIn(axes_negative_bd),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_negative_bd, Gather7LayerTest, gather7ParamsSubset_negative_bd, Gather7LayerTest::getTestCaseName);


///// GATHER-8 /////

const std::vector<std::vector<ov::Shape>> data_shapes_4d_gather8 = {
        {{10, 3, 1, 2}},
        {{10, 3, 3, 1}},
        {{10, 2, 2, 7}},
        {{10, 2, 2, 2}},
        {{10, 3, 4, 4}},
        {{10, 2, 3, 17}}
};
const std::vector<ov::Shape> idx_shapes_4d_gather8 = {
        {10, 1, 1},
        {10, 1, 2},
        {10, 1, 3},
        {10, 2, 2},
        {10, 1, 7},
        {10, 2, 4},
        {10, 3, 3},
        {10, 3, 5},
        {10, 7, 3},
        {10, 8, 7}
};
const std::vector<std::tuple<int, int>> axes_batches_4d_gather8 = {
        {3, 0},
        {-1, -2},
        {2, -3},
        {2, 1},
        {1, 0},
        {1, 1},
        {0, 0}
};

INSTANTIATE_TEST_SUITE_P(smoke_static_4D, Gather8LayerTest,
        testing::Combine(
                testing::ValuesIn(ov::test::static_shapes_to_test_representation(data_shapes_4d_gather8)),
                testing::ValuesIn(idx_shapes_4d_gather8),
                testing::ValuesIn(axes_batches_4d_gather8),
                testing::ValuesIn(model_types),
                testing::Values(ov::test::utils::DEVICE_CPU)),
        Gather8LayerTest::getTestCaseName);


const std::vector<std::vector<ov::Shape>> data_shapes_vec2_gather8 = {
        {{5, 4}},
        {{11, 4}},
        {{23, 4}},
        {{35, 4}},
        {{51, 4}},
        {{71, 4}}
};
const std::vector<ov::Shape> idx_shapes_vec2_gather8 = {{1}};

const std::vector<std::tuple<int, int>> axes_batches_vec2_gather8 = {{1, 0}};

const auto gatherParamsVec2 = testing::Combine(
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(data_shapes_vec2_gather8)),
        testing::ValuesIn(idx_shapes_vec2_gather8),
        testing::ValuesIn(axes_batches_vec2_gather8),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Vec2, Gather8LayerTest, gatherParamsVec2, Gather8LayerTest::getTestCaseName);


const std::vector<ov::Shape> data_shapes_vec3_gather8 = {{4, 4}};
const std::vector<ov::Shape> idx_shapes_vec3_gather8 = {{5}, {11}, {21}, {35}, {55}, {70}};

const std::vector<std::tuple<int, int>> axes_batches_vec3_gather8 = {{1, 0}};

const auto gatherParamsVec3 = testing::Combine(
        testing::Values(ov::test::static_shapes_to_test_representation(data_shapes_vec3_gather8)),
        testing::ValuesIn(idx_shapes_vec3_gather8),
        testing::ValuesIn(axes_batches_vec3_gather8),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Vec3, Gather8LayerTest, gatherParamsVec3, Gather8LayerTest::getTestCaseName);


const ov::test::gather7ParamsTuple dummyParams = {
        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{2, 3}}), // input shape
        ov::Shape{2, 2},                // indices shape
        std::tuple<int, int>{1, 1},     // axis, batch
        ov::element::f32,               // model type
        ov::test::utils::DEVICE_CPU     // device
};

const std::vector<std::vector<int64_t>> indicesData = {
        {0, 1, 2, 0},           // positive in bound
        {-1, -2, -3, -1},       // negative in bound
        {-1, 0, 1, 2},          // positive and negative in bound
        {0, 1, 2, 3},           // positive out of bound
        {-1, -2, -3, -4},       // negative out of bound
        {0, 4, -4, 0},          // positive and negative out of bound
};

const auto gatherWithIndicesParams = testing::Combine(
        testing::Values(dummyParams),
        testing::ValuesIn(indicesData)
);

INSTANTIATE_TEST_SUITE_P(smoke, Gather8withIndicesDataLayerTest, gatherWithIndicesParams, Gather8withIndicesDataLayerTest::getTestCaseName);

std::vector<ov::test::GatherStringParamsTuple> string_cases_params{
    {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{3}}),  // input shape
     ov::Shape{1},                                                                 // indices shape
     std::tuple<int, int>{0, 0},                                                   // axis, batch
     ov::element::string,                                                          // model type
     ov::test::utils::DEVICE_CPU,                                                  // device
     std::vector<int64_t>{0},                                                      // indices value
     std::vector<std::string>{"Abc", "xyz", "..."}},                               // data str value
    {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{3}}),
     ov::Shape{1},
     std::tuple<int, int>{0, 0},
     ov::element::string,
     ov::test::utils::DEVICE_CPU,
     std::vector<int64_t>{1},
     std::vector<std::string>{"Abc", "xyz", "..."}},
    {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{3}}),
     ov::Shape{2},
     std::tuple<int, int>{0, 0},
     ov::element::string,
     ov::test::utils::DEVICE_CPU,
     std::vector<int64_t>{0, 2},
     std::vector<std::string>{"Abc", "xyz", "..."}},
    {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{3}}),
     ov::Shape{2},
     std::tuple<int, int>{0, 0},
     ov::element::string,
     ov::test::utils::DEVICE_CPU,
     std::vector<int64_t>{0, 1},
     std::vector<std::string>{"Ab", "1345", "xyz"}},
    {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{2, 2}}),
     ov::Shape{1},
     std::tuple<int, int>{0, 0},
     ov::element::string,
     ov::test::utils::DEVICE_CPU,
     std::vector<int64_t>{1},
     std::vector<std::string>{"A", "B c", "d.Ef", " G h,i;"}},
    {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{2, 2, 2}}),
     ov::Shape{1},
     std::tuple<int, int>{0, 0},
     ov::element::string,
     ov::test::utils::DEVICE_CPU,
     std::vector<int64_t>{1},
     std::vector<std::string>{"A", "B c", "d.Ef", " G h,i;", "JK ", "l,m,n,", " ", " \0"}},
    {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{2, 1, 2}}),
     ov::Shape{2, 1, 2},
     std::tuple<int, int>{2, 2},
     ov::element::string,
     ov::test::utils::DEVICE_CPU,
     std::vector<int64_t>{0, 1, 1, 0},
     std::vector<std::string>{"A", "B c", "d.Ef", " G h,i;"}}};

const auto gatherWithStringParams = testing::ValuesIn(string_cases_params);

INSTANTIATE_TEST_CASE_P(smoke_gather_string,
                        GatherStringWithIndicesDataLayerTest,
                        gatherWithStringParams,
                        GatherStringWithIndicesDataLayerTest::getTestCaseName);

}  // namespace
