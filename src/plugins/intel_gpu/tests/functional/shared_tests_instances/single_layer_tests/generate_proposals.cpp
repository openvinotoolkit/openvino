// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/generate_proposals.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace ov::test;
using namespace ov::test::subgraph;

namespace {

const std::vector<float> min_size = { 1.0f, 0.0f };
const std::vector<float> nms_threshold = { 0.7f };
const std::vector<int64_t> post_nms_count = { 6 };
const std::vector<int64_t> pre_nms_count = { 14, 1000 };

template <typename T>
const std::vector<std::pair<std::string, std::vector<ov::Tensor>>> getInputTensors() {
    const std::vector<std::pair<std::string, std::vector<ov::Tensor>>> input_tensors = {
        {
            "empty",
            {
                // 3
                ov::test::utils::create_tensor<T>(ov::element::from<T>(), ov::Shape{2, 3}, {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f}),
                // 2 x 6 x 3 x 4 = 144
                ov::test::utils::create_tensor<T>(ov::element::from<T>(), ov::Shape{2, 6, 3, 4}, std::vector<T>(144, 1.0f)),
                // 2 x 12 x 2 x 6 = 144 * 2
                ov::test::utils::create_tensor<T>(ov::element::from<T>(), ov::Shape{2, 12, 2, 6}, std::vector<T>(288, 1.0f)),
                // {2 x 3 x 2 x 6} = 36 * 2
                ov::test::utils::create_tensor<T>(ov::element::from<T>(), ov::Shape{2, 3, 2, 6}, {
                        5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f,
                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f,
                        5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f,
                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f})
            }
    },
    {
        "filled2",
        {
            ov::test::utils::create_tensor<T>(ov::element::from<T>(), ov::Shape{2, 3}, {200.0f, 200.0f, 4.0f, 200.0f, 200.0f, 4.0f}),
            ov::test::utils::create_tensor<T>(ov::element::from<T>(), ov::Shape{2, 6, 3, 4}, {0.0f,  1.0f,  2.0f,  3.0f,
                                                                                            4.0f,  5.0f,  6.0f,  7.0f,
                                                                                            8.0f,  9.0f, 10.0f, 11.0f,
                                                                                            12.0f, 13.0f, 14.0f, 15.0f,
                                                                                            16.0f, 17.0f, 18.0f, 19.0f,
                                                                                            20.0f, 21.0f, 22.0f, 23.0f,
                                                                                            24.0f, 25.0f, 26.0f, 27.0f,
                                                                                            28.0f, 29.0f, 30.0f, 31.0f,
                                                                                            32.0f, 33.0f, 34.0f, 35.0f,
                                                                                            36.0f, 37.0f, 38.0f, 39.0f,
                                                                                            40.0f, 41.0f, 42.0f, 43.0f,
                                                                                            44.0f, 45.0f, 46.0f, 47.0f,
                                                                                            48.0f, 49.0f, 50.0f, 51.0f,
                                                                                            52.0f, 53.0f, 54.0f, 55.0f,
                                                                                            56.0f, 57.0f, 58.0f, 59.0f,
                                                                                            60.0f, 61.0f, 62.0f, 63.0f,
                                                                                            64.0f, 65.0f, 66.0f, 67.0f,
                                                                                            68.0f, 69.0f, 70.0f, 71.0f,
                                                                                            72.0f, 73.0f, 74.0f, 75.0f,
                                                                                            76.0f, 77.0f, 78.0f, 79.0f,
                                                                                            80.0f, 81.0f, 82.0f, 83.0f,
                                                                                            84.0f, 85.0f, 86.0f, 87.0f,
                                                                                            88.0f, 89.0f, 90.0f, 91.0f,
                                                                                            92.0f, 93.0f, 94.0f, 95.0f,
                                                                                            96.0f,  97.0f,  98.0f,  99.0f,
                                                                                            100.0f, 101.0f, 102.0f, 103.0f,
                                                                                            104.0f, 105.0f, 106.0f, 107.0f,
                                                                                            108.0f, 109.0f, 110.0f, 111.0f,
                                                                                            112.0f, 113.0f, 114.0f, 115.0f,
                                                                                            116.0f, 117.0f, 118.0f, 119.0f,
                                                                                            120.0f, 121.0f, 122.0f, 123.0f,
                                                                                            124.0f, 125.0f, 126.0f, 127.0f,
                                                                                            128.0f, 129.0f, 130.0f, 131.0f,
                                                                                            132.0f, 133.0f, 134.0f, 135.0f,
                                                                                            136.0f, 137.0f, 138.0f, 139.0f,
                                                                                            140.0f, 141.0f, 142.0f, 143.0f}),
            ov::test::utils::create_tensor<T>(ov::element::from<T>(), ov::Shape{2, 12, 2, 6}, {
                    0.5337073f,  0.86607957f, 0.55151343f, 0.21626699f, 0.4462629f,  0.03985678f,
                    0.5157072f,  0.9932138f,  0.7565954f,  0.43803605f, 0.802818f,   0.14834064f,
                    0.53932905f, 0.14314f,    0.3817048f,  0.95075196f, 0.05516243f, 0.2567484f,
                    0.25508744f, 0.77438325f, 0.43561f,    0.2094628f,  0.8299043f,  0.44982538f,
                    0.95615596f, 0.5651084f,  0.11801951f, 0.05352486f, 0.9774733f,  0.14439464f,
                    0.62644225f, 0.14370479f, 0.54161614f, 0.557915f,   0.53102225f, 0.0840179f,
                    0.7249888f,  0.9843559f,  0.5490522f,  0.53788143f, 0.822474f,   0.3278008f,
                    0.39688024f, 0.3286012f,  0.5117038f,  0.04743988f, 0.9408995f,  0.29885054f,
                    0.81039643f, 0.85277915f, 0.06807619f, 0.86430097f, 0.36225632f, 0.16606331f,
                    0.5401001f,  0.7541649f,  0.11998601f, 0.5131829f,  0.40606487f, 0.327888f,
                    0.27721855f, 0.6378373f,  0.22795396f, 0.4961256f,  0.3215895f,  0.15607187f,
                    0.14782153f, 0.8908137f,  0.8835288f,  0.834191f,   0.29907143f, 0.7983525f,
                    0.755875f,   0.30837986f, 0.0839176f,  0.26624718f, 0.04371626f, 0.09472824f,
                    0.20689541f, 0.37622106f, 0.1083321f,  0.1342548f,  0.05815459f, 0.7676379f,
                    0.8105144f,  0.92348766f, 0.26761323f, 0.7183306f,  0.8947588f,  0.19020908f,
                    0.42731014f, 0.7473663f,  0.85775334f, 0.9340091f,  0.3278848f,  0.755993f,
                    0.05307213f, 0.39705503f, 0.21003333f, 0.5625373f,  0.66188884f, 0.80521655f,
                    0.6125863f,  0.44678232f, 0.97802377f, 0.0204936f,  0.02686367f, 0.7390654f,
                    0.74631f,    0.58399844f, 0.5988792f,  0.37413648f, 0.5946692f,  0.6955776f,
                    0.36377597f, 0.7891322f,  0.40900692f, 0.99139464f, 0.50169915f, 0.41435778f,
                    0.17142445f, 0.26761186f, 0.31591868f, 0.14249913f, 0.12919712f, 0.5418711f,
                    0.6523203f,  0.50259084f, 0.7379765f,  0.01171071f, 0.94423133f, 0.00841132f,
                    0.97486794f, 0.2921785f,  0.7633071f,  0.88477814f, 0.03563205f, 0.50833166f,
                    0.01354555f, 0.535081f,   0.41366324f, 0.0694767f,  0.9944055f,  0.9981207f,
                    0.5337073f,  0.86607957f, 0.55151343f, 0.21626699f, 0.4462629f,  0.03985678f,
                    0.5157072f,  0.9932138f,  0.7565954f,  0.43803605f, 0.802818f,   0.14834064f,
                    0.53932905f, 0.14314f,    0.3817048f,  0.95075196f, 0.05516243f, 0.2567484f,
                    0.25508744f, 0.77438325f, 0.43561f,    0.2094628f,  0.8299043f,  0.44982538f,
                    0.95615596f, 0.5651084f,  0.11801951f, 0.05352486f, 0.9774733f,  0.14439464f,
                    0.62644225f, 0.14370479f, 0.54161614f, 0.557915f,   0.53102225f, 0.0840179f,
                    0.7249888f,  0.9843559f,  0.5490522f,  0.53788143f, 0.822474f,   0.3278008f,
                    0.39688024f, 0.3286012f,  0.5117038f,  0.04743988f, 0.9408995f,  0.29885054f,
                    0.81039643f, 0.85277915f, 0.06807619f, 0.86430097f, 0.36225632f, 0.16606331f,
                    0.5401001f,  0.7541649f,  0.11998601f, 0.5131829f,  0.40606487f, 0.327888f,
                    0.27721855f, 0.6378373f,  0.22795396f, 0.4961256f,  0.3215895f,  0.15607187f,
                    0.14782153f, 0.8908137f,  0.8835288f,  0.834191f,   0.29907143f, 0.7983525f,
                    0.755875f,   0.30837986f, 0.0839176f,  0.26624718f, 0.04371626f, 0.09472824f,
                    0.20689541f, 0.37622106f, 0.1083321f,  0.1342548f,  0.05815459f, 0.7676379f,
                    0.8105144f,  0.92348766f, 0.26761323f, 0.7183306f,  0.8947588f,  0.19020908f,
                    0.42731014f, 0.7473663f,  0.85775334f, 0.9340091f,  0.3278848f,  0.755993f,
                    0.05307213f, 0.39705503f, 0.21003333f, 0.5625373f,  0.66188884f, 0.80521655f,
                    0.6125863f,  0.44678232f, 0.97802377f, 0.0204936f,  0.02686367f, 0.7390654f,
                    0.74631f,    0.58399844f, 0.5988792f,  0.37413648f, 0.5946692f,  0.6955776f,
                    0.36377597f, 0.7891322f,  0.40900692f, 0.99139464f, 0.50169915f, 0.41435778f,
                    0.17142445f, 0.26761186f, 0.31591868f, 0.14249913f, 0.12919712f, 0.5418711f,
                    0.6523203f,  0.50259084f, 0.7379765f,  0.01171071f, 0.94423133f, 0.00841132f,
                    0.97486794f, 0.2921785f,  0.7633071f,  0.88477814f, 0.03563205f, 0.50833166f,
                    0.01354555f, 0.535081f,   0.41366324f, 0.0694767f,  0.9944055f,  0.9981207f}),
            ov::test::utils::create_tensor<T>(ov::element::from<T>(), ov::Shape{2, 3, 2, 6}, {
                    0.56637216f, 0.90457034f, 0.69827306f, 0.4353543f,  0.47985056f, 0.42658508f,
                    0.14516132f, 0.08081771f, 0.1799732f,  0.9229515f,  0.42420176f, 0.50857586f,
                    0.82664067f, 0.4972319f,  0.3752427f,  0.56731623f, 0.18241242f, 0.33252355f,
                    0.30608943f, 0.6572437f,  0.69185436f, 0.88646156f, 0.36985755f, 0.5590753f,
                    0.5256446f,  0.03342898f, 0.1344396f,  0.68642473f, 0.37953874f, 0.32575172f,
                    0.21108444f, 0.5661886f,  0.45378175f, 0.62126315f, 0.26799858f, 0.37272978f,
                    0.56637216f, 0.90457034f, 0.69827306f, 0.4353543f,  0.47985056f, 0.42658508f,
                    0.14516132f, 0.08081771f, 0.1799732f,  0.9229515f,  0.42420176f, 0.50857586f,
                    0.82664067f, 0.4972319f,  0.3752427f,  0.56731623f, 0.18241242f, 0.33252355f,
                    0.30608943f, 0.6572437f,  0.69185436f, 0.88646156f, 0.36985755f, 0.5590753f,
                    0.5256446f,  0.03342898f, 0.1344396f,  0.68642473f, 0.37953874f, 0.32575172f,
                    0.21108444f, 0.5661886f,  0.45378175f, 0.62126315f, 0.26799858f, 0.37272978f}),
            }
        }
    };
    return input_tensors;
}

constexpr size_t num_batches = 2;
constexpr size_t height = 2;
constexpr size_t width = 6;
constexpr size_t number_of_anchors = 3;

const std::vector<std::vector<InputShape>> input_shape = {
        // im_info / anchors / boxesdeltas / scores
        static_shapes_to_test_representation({{num_batches, 3},
                                              {height, width, number_of_anchors, 4},
                                              {num_batches, number_of_anchors * 4, height, width},
                                              {num_batches, number_of_anchors, height, width}}),
};



INSTANTIATE_TEST_SUITE_P(
        smoke_GenerateProposalsLayerTest_f16,
        GenerateProposalsLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(input_shape),
                ::testing::ValuesIn(min_size),
                ::testing::ValuesIn(nms_threshold),
                ::testing::ValuesIn(post_nms_count),
                ::testing::ValuesIn(pre_nms_count),
                ::testing::ValuesIn({true}),
                ::testing::ValuesIn(getInputTensors<ov::float16>()),
                ::testing::ValuesIn({ov::element::Type_t::f16}),
                ::testing::ValuesIn({ov::element::Type_t::i32, ov::element::Type_t::i64}),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
        GenerateProposalsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_GenerateProposalsLayerTest_f32,
        GenerateProposalsLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(input_shape),
                ::testing::ValuesIn(min_size),
                ::testing::ValuesIn(nms_threshold),
                ::testing::ValuesIn(post_nms_count),
                ::testing::ValuesIn(pre_nms_count),
                ::testing::ValuesIn({false}),
                ::testing::ValuesIn(getInputTensors<float>()),
                ::testing::ValuesIn({ov::element::Type_t::f32}),
                ::testing::ValuesIn({ov::element::Type_t::i32, ov::element::Type_t::i64}),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
        GenerateProposalsLayerTest::getTestCaseName);

} // namespace
