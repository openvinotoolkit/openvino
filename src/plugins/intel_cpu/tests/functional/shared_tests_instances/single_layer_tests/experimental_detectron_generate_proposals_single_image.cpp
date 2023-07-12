// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/experimental_detectron_generate_proposals_single_image.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace ov::test;
using namespace ov::test::subgraph;

namespace {

const std::vector<float> min_size = { 0 };
const std::vector<float> nms_threshold = { 0.699999988079071 };
const std::vector<int64_t> post_nms_count = { 6 };
const std::vector<int64_t> pre_nms_count = { 1000 };

const std::vector<std::pair<std::string, std::vector<ov::Tensor>>> inputTensors = {
    {
        "empty",
        {
            // 3
            ov::test::utils::create_tensor<float>(ov::element::f32, ov::Shape{3}, {1.0f, 1.0f, 1.0f}),
            // 36 x 4 = 144
            ov::test::utils::create_tensor<float>(ov::element::f32, ov::Shape{36, 4}, {
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}),
            // 12 x 2 x 6 = 144
            ov::test::utils::create_tensor<float>(ov::element::f32, ov::Shape{12, 2, 6}, {
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}),
            // {3 x 2 x 6} = 36
            ov::test::utils::create_tensor<float>(ov::element::f32, ov::Shape{3, 2, 6}, {
                5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f})
        }
    },
    {
        "filled",
        {
            ov::test::utils::create_tensor<float>(ov::element::f32, ov::Shape{3}, {150.0, 150.0, 1.0}),
            ov::test::utils::create_tensor<float>(ov::element::f32, ov::Shape{36, 4}, {
                12.0f, 68.0f, 102.0f, 123.0f, 46.0f, 80.0f,  79.0f,  128.0f, 33.0f, 71.0f, 127.0f, 86.0f,  33.0f, 56.0f, 150.0f, 73.0f,
                5.0f,  41.0f, 93.0f,  150.0f, 74.0f, 66.0f,  106.0f, 115.0f, 17.0f, 37.0f, 87.0f,  150.0f, 31.0f, 27.0f, 150.0f, 39.0f,
                29.0f, 23.0f, 112.0f, 123.0f, 41.0f, 37.0f,  103.0f, 150.0f, 8.0f,  46.0f, 98.0f,  111.0f, 7.0f,  69.0f, 114.0f, 150.0f,
                70.0f, 21.0f, 150.0f, 125.0f, 54.0f, 19.0f,  132.0f, 68.0f,  62.0f, 8.0f,  150.0f, 101.0f, 57.0f, 81.0f, 150.0f, 97.0f,
                79.0f, 29.0f, 109.0f, 130.0f, 12.0f, 63.0f,  100.0f, 150.0f, 17.0f, 33.0f, 113.0f, 150.0f, 90.0f, 78.0f, 150.0f, 111.0f,
                47.0f, 68.0f, 150.0f, 71.0f,  66.0f, 103.0f, 111.0f, 150.0f, 4.0f,  17.0f, 112.0f, 94.0f,  12.0f, 8.0f,  119.0f, 98.0f,
                54.0f, 56.0f, 120.0f, 150.0f, 56.0f, 29.0f,  150.0f, 31.0f,  42.0f, 3.0f,  139.0f, 92.0f,  41.0f, 65.0f, 150.0f, 130.0f,
                49.0f, 13.0f, 143.0f, 30.0f,  40.0f, 60.0f,  150.0f, 150.0f, 23.0f, 73.0f, 24.0f,  115.0f, 56.0f, 84.0f, 107.0f, 108.0f,
                63.0f, 8.0f,  142.0f, 125.0f, 78.0f, 37.0f,  93.0f,  144.0f, 40.0f, 34.0f, 150.0f, 46.0f,  30.0f, 21.0f, 150.0f, 120.0f}),
            ov::test::utils::create_tensor<float>(ov::element::f32, ov::Shape{12, 2, 6}, {
                9.062256f,   10.883133f, 9.8441105f,   12.694285f,   0.41781136f, 8.749107f,      14.990341f,  6.587644f,  1.4206103f,
                13.299262f,  12.432549f, 2.736371f,    0.22732796f,  6.3361835f,  12.268727f,     2.1009045f,  4.771589f,  2.5131326f,
                5.610736f,   9.3604145f, 4.27379f,     8.317948f,    0.60510135f, 6.7446275f,     1.0207708f,  1.1352817f, 1.5785321f,
                1.718335f,   1.8093798f, 0.99247587f,  1.3233583f,   1.7432803f,  1.8534478f,     1.2593061f,  1.7394226f, 1.7686696f,
                1.647999f,   1.7611449f, 1.3119122f,   0.03007332f,  1.1106564f,  0.55669737f,    0.2546148f,  1.9181818f, 0.7134989f,
                2.0407224f,  1.7211134f, 1.8565536f,   14.562747f,   2.8786168f,  0.5927796f,     0.2064463f,  7.6794515f, 8.672126f,
                10.139171f,  8.002429f,  7.002932f,    12.6314945f,  10.550842f,  0.15784842f,    0.3194304f,  10.752157f, 3.709805f,
                11.628928f,  0.7136225f, 14.619964f,   15.177284f,   2.2824087f,  15.381494f,     0.16618137f, 7.507227f,  11.173228f,
                0.4923559f,  1.8227729f, 1.4749299f,   1.7833921f,   1.2363617f,  -0.23659119f,   1.5737582f,  1.779316f,  1.9828427f,
                1.0482665f,  1.4900246f, 1.3563544f,   1.5341306f,   0.7634312f,  4.6216766e-05f, 1.6161222f,  1.7512476f, 1.9363779f,
                0.9195784f,  1.4906164f, -0.03244795f, 0.681073f,    0.6192401f,  1.8033613f,     14.146055f,  3.4043705f, 15.292292f,
                3.5295358f,  11.138999f, 9.952057f,    5.633434f,    12.114562f,  9.427372f,      12.384038f,  9.583308f,  8.427233f,
                15.293704f,  3.288159f,  11.64898f,    9.350885f,    2.0037227f,  13.523184f,     4.4176426f,  6.1057625f, 14.400079f,
                8.248259f,   11.815807f, 15.713364f,   1.0023532f,   1.3203261f,  1.7100681f,     0.7407832f,  1.09448f,   1.7188418f,
                1.4412547f,  1.4862992f, 0.74790007f,  0.31571656f,  0.6398838f,  2.0236106f,     1.1869069f,  1.7265586f, 1.2624544f,
                0.09934269f, 1.3508598f, 0.85212964f,  -0.38968498f, 1.7059708f,  1.6533034f,     1.7400402f,  1.8123854f, -0.43063712f}),
            ov::test::utils::create_tensor<float>(ov::element::f32, ov::Shape{3, 2, 6}, {
                0.7719922f,  0.35906568f, 0.29054508f, 0.18124384f, 0.5604661f,  0.84750974f, 0.98948747f, 0.009793862f, 0.7184191f,
                0.5560748f,  0.6952493f,  0.6732593f,  0.3306898f,  0.6790913f,  0.41128764f, 0.34593266f, 0.94296855f,  0.7348507f,
                0.24478768f, 0.94024557f, 0.05405676f, 0.06466125f, 0.36244348f, 0.07942984f, 0.10619422f, 0.09412837f,  0.9053611f,
                0.22870538f, 0.9237487f,  0.20986171f, 0.5067282f,  0.29709867f, 0.53138554f, 0.189101f,   0.4786443f,   0.88421875f}),
        }
    }
};

const std::vector<std::vector<InputShape>> dynamicInputShape = {
    // im_info / anchors / deltas / scores
    static_shapes_to_test_representation({{3}, {36, 4}, {12, 2, 6}, {3, 2, 6}}),
    {
        {{-1}, {{3}}},
        {{-1, -1}, {{36, 4}}},
        {{-1, -1, -1}, {{12, 2, 6}}},
        {{-1, -1, -1}, {{3, 2, 6}}}
    },
    {
        {{{3, 6}}, {{3}}},
        {{{36, 72}, {4, 8}}, {{36, 4}}},
        {{{12, 24}, {2, 4}, {6, 12}}, {{12, 2, 6}}},
        {{{3, 6}, {2, 4}, {6, 12}}, {{3, 2, 6}}}
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_ExperimentalDetectronGenerateProposalsSingleImageLayerTest,
    ExperimentalDetectronGenerateProposalsSingleImageLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShape),
        ::testing::ValuesIn(min_size),
        ::testing::ValuesIn(nms_threshold),
        ::testing::ValuesIn(post_nms_count),
        ::testing::ValuesIn(pre_nms_count),
        ::testing::ValuesIn(inputTensors),
        ::testing::Values(ov::element::Type_t::f32),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    ExperimentalDetectronGenerateProposalsSingleImageLayerTest::getTestCaseName);
} // namespace
