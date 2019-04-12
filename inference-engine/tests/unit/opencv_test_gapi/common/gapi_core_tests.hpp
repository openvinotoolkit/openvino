// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENCV_GAPI_CORE_TESTS_HPP
#define OPENCV_GAPI_CORE_TESTS_HPP

#include "gapi_tests_common.hpp"
#include "ie_preprocess.hpp"

#include <gtest/gtest.h>

namespace opencv_test
{

struct ResizeTestGAPI: public testing::TestWithParam<std::tuple<int, int, std::pair<cv::Size, cv::Size>, double>> {};
struct SplitTestGAPI: public TestParams<std::tuple<int, int, cv::Size>> {};
struct MergeTestGAPI: public TestParams<std::tuple<int, int, cv::Size>> {};

//------------------------------------------------------------------------------

struct ResizeTestIE: public testing::TestWithParam<std::tuple<int, int, std::pair<cv::Size, cv::Size>, double>> {};

struct SplitTestIE: public TestParams<std::tuple<int, cv::Size>> {};
struct MergeTestIE: public TestParams<std::tuple<int, cv::Size>> {};

//------------------------------------------------------------------------------

using PreprocParams = std::tuple< InferenceEngine::Precision     // input-output data type
                                , InferenceEngine::ResizeAlgorithm // resize algorithm, if needed
                                , InferenceEngine::Layout        // input tensor layout
                                , InferenceEngine::Layout        // output tensor layout
                                , int                            // number of channels
                                , std::pair<cv::Size, cv::Size>
                                >;

struct PreprocTest: public TestParams<PreprocParams> {};

} // opencv_test

#endif //OPENCV_GAPI_CORE_TESTS_HPP
