// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/data_utils.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <single_layer_tests/experimental_detectron_prior_grid_generator.hpp>

namespace {

const std::initializer_list<ov::test::subgraph::ExperimentalDetectronPriorGridGeneratorTestParam> params{
    // flatten = true (output tensor is 2D)
    {{true, 0, 0, 4.0f, 4.0f},
     ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 4, 5}, {1, 3, 100, 200}})},
    // task #72587
    //{
    //        {true, 3, 6, 64.0f, 64.0f},
    //        ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 100, 100}, {1, 3, 100, 200}})
    //},
    //// flatten = false (output tensor is 4D)
    {{false, 0, 0, 8.0f, 8.0f},
     ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 3, 7}, {1, 3, 100, 200}})},
    // task #72587
    //{
    //        {false, 5, 3, 32.0f, 32.0f},
    //        ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 100, 100}, {1, 3, 100, 200}})
    //},
};

template <typename T>
std::vector<std::pair<std::string, std::vector<ov::Tensor>>> getInputTensors() {
    std::vector<std::pair<std::string, std::vector<ov::Tensor>>> tensors{
        {"test#1",
         {ov::test::utils::create_tensor<T>(
             ov::element::from<T>(),
             {3, 4},
             std::vector<T>{-24.5, -12.5, 24.5, 12.5, -16.5, -16.5, 16.5, 16.5, -12.5, -24.5, 12.5, 24.5})}},
        {"test#2",
         {ov::test::utils::create_tensor<T>(
             ov::element::from<T>(),
             {3, 4},
             std::vector<T>{-44.5, -24.5, 44.5, 24.5, -32.5, -32.5, 32.5, 32.5, -24.5, -44.5, 24.5, 44.5})}},
        {"test#3",
         {ov::test::utils::create_tensor<T>(
             ov::element::from<T>(),
             {3, 4},
             std::
                 vector<T>{-364.5, -184.5, 364.5, 184.5, -256.5, -256.5, 256.5, 256.5, -180.5, -360.5, 180.5, 360.5})}},
        {"test#4",
         {ov::test::utils::create_tensor<T>(
             ov::element::from<T>(),
             {3, 4},
             std::vector<T>{-180.5, -88.5, 180.5, 88.5, -128.5, -128.5, 128.5, 128.5, -92.5, -184.5, 92.5, 184.5})}}};
    return tensors;
}

using ov::test::subgraph::ExperimentalDetectronPriorGridGeneratorLayerTest;

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronPriorGridGenerator_f32,
                         ExperimentalDetectronPriorGridGeneratorLayerTest,
                         testing::Combine(testing::ValuesIn(params),
                                          testing::ValuesIn(getInputTensors<float>()),
                                          testing::ValuesIn({ov::element::Type_t::f32}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronPriorGridGeneratorLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronPriorGridGenerator_f16,
                         ExperimentalDetectronPriorGridGeneratorLayerTest,
                         testing::Combine(testing::ValuesIn(params),
                                          testing::ValuesIn(getInputTensors<ov::float16>()),
                                          testing::ValuesIn({ov::element::Type_t::f16}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronPriorGridGeneratorLayerTest::getTestCaseName);
}  // namespace
