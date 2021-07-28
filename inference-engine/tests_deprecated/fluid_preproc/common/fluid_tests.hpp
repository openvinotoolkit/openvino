// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FLUID_TESTS_HPP
#define FLUID_TESTS_HPP

#include "fluid_tests_common.hpp"
#include "ie_preprocess.hpp"

#include <gtest/gtest.h>

struct ResizeTestGAPI: public testing::TestWithParam<std::tuple<int, int, std::pair<cv::Size, cv::Size>, double>> {};
struct ResizeRGB8UTestGAPI: public testing::TestWithParam<std::tuple<int, int, std::pair<cv::Size, cv::Size>, double>> {};
struct SplitTestGAPI: public TestParams<std::tuple<int, int, cv::Size, double>> {};
struct ChanToPlaneTestGAPI: public TestParams<std::tuple<int, int, cv::Size, double>> {};
struct MergeTestGAPI: public TestParams<std::tuple<int, int, cv::Size, double>> {};
struct NV12toRGBTestGAPI: public TestParams<std::tuple<cv::Size, double>> {};
struct I420toRGBTestGAPI: public TestParams<std::tuple<cv::Size, double>> {};
struct ResizeRoiTestGAPI: public testing::TestWithParam<std::tuple<int, int, std::pair<cv::Size, cv::Size>, cv::Rect, double>> {};
struct ResizeRGB8URoiTestGAPI: public testing::TestWithParam<std::tuple<int, int, std::pair<cv::Size, cv::Size>, cv::Rect, double>> {};
struct ConvertDepthTestGAPI: public TestParams<std::tuple<
                            int,  // input matrix depth
                            int,  // output matrix depth
                            cv::Size,
                            double>>   // tolerance
{};
struct DivCTestGAPI: public TestParams<std::tuple<
                            int,  // input matrix depth
                            int,  // input matrix channels number
                            cv::Size,
                            cv::Scalar, // second operarnd
                            double>>    // tolerance
{};

struct SubCTestGAPI : public DivCTestGAPI
{};

struct MeanValueGAPI: public TestParams<std::tuple<cv::Size, double>> {};
//------------------------------------------------------------------------------

struct ResizeTestIE: public testing::TestWithParam<std::tuple<int, int, std::pair<cv::Size, cv::Size>, double>> {};

struct SplitTestIE: public TestParams<std::tuple<int, cv::Size, double>> {};
struct MergeTestIE: public TestParams<std::tuple<int, cv::Size, double>> {};

struct ColorConvertTestIE:
    public testing::TestWithParam<std::tuple<int,  // matrix depth
                                             InferenceEngine::ColorFormat,  // input color format
                                             InferenceEngine::Layout,  // input layout
                                             InferenceEngine::Layout,  // output layout
                                             cv::Size,  // matrix size (input and output)
                                             double>>  // tolerance
{};

struct ColorConvertYUV420TestIE:
    public testing::TestWithParam<std::tuple<InferenceEngine::ColorFormat,  // input color format NV12 or I420
                                             InferenceEngine::Layout,       // output layout
                                             cv::Size,                      // matrix size (input and output)
                                             double>>                       // tolerance
{};

struct PrecisionConvertTestIE: public TestParams<std::tuple<cv::Size,
                                                            int,     // input  matrix depth
                                                            int,     // output matrix depth
                                                            double>> // tolerance
{};

//------------------------------------------------------------------------------

using PreprocParams = std::tuple< std::pair<InferenceEngine::Precision     // input data type
                                           , InferenceEngine::Precision>   // output data type
                                , InferenceEngine::ResizeAlgorithm // resize algorithm, if needed
                                , InferenceEngine::ColorFormat // input color format, if needed
                                , InferenceEngine::Layout        // input tensor layout
                                , InferenceEngine::Layout        // output tensor layout
                                , std::pair<int, int>   // number of input and output channels
                                , std::pair<cv::Size, cv::Size>
                                >;

struct PreprocTest: public TestParams<PreprocParams> {};

#endif //FLUID_TESTS_HPP
