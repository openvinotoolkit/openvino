// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gapi_core_tests.hpp"

#include "ie_preprocess_gapi_kernels.hpp"

#include <opencv2/opencv.hpp>

#include <gtest/gtest.h>

namespace opencv_test
{

#define CORE_FLUID InferenceEngine::gapi::preprocKernels()

#define TEST_SIZES        \
    cv::Size(3840, 2160), \
    cv::Size(1920, 1080), \
    cv::Size(1280,  720), \
    cv::Size(1280,  960), \
    cv::Size( 960,  720), \
    cv::Size( 640,  480), \
    cv::Size( 320,  200), \
    cv::Size( 113,   71)

#define TEST_RESIZE_DOWN \
    std::make_pair(cv::Size(3840, 2160), cv::Size(1920, 1080)), \
    std::make_pair(cv::Size(3840, 2160), cv::Size(1280,  720)), \
    std::make_pair(cv::Size(1920, 1080), cv::Size(1280,  720)), \
    std::make_pair(cv::Size(1920, 1080), cv::Size( 640,  480)), \
    std::make_pair(cv::Size(1280,  720), cv::Size( 640,  480)), \
    std::make_pair(cv::Size(1280,  720), cv::Size( 320,  200)), \
    std::make_pair(cv::Size( 640,  480), cv::Size( 320,  200)), \
    std::make_pair(cv::Size( 640,  480), cv::Size( 113,   71)), \
    std::make_pair(cv::Size( 320,  200), cv::Size( 113,   71))

#define TEST_RESIZE_UP \
    std::make_pair(cv::Size(1920, 1080), cv::Size(3840, 2160)), \
    std::make_pair(cv::Size(1280,  720), cv::Size(3840, 2160)), \
    std::make_pair(cv::Size(1280,  720), cv::Size(1920, 1080)), \
    std::make_pair(cv::Size( 640,  480), cv::Size(1920, 1080)), \
    std::make_pair(cv::Size( 640,  480), cv::Size(1280,  720)), \
    std::make_pair(cv::Size( 320,  200), cv::Size(1280,  720)), \
    std::make_pair(cv::Size( 320,  200), cv::Size( 640,  480)), \
    std::make_pair(cv::Size( 113,   71), cv::Size( 640,  480)), \
    std::make_pair(cv::Size( 113,   71), cv::Size( 320,  200))

#define TEST_RESIZE_HORZ \
    std::make_pair(cv::Size(3840, 2160), cv::Size(1920, 2160)), \
    std::make_pair(cv::Size(1920, 1080), cv::Size(3840, 1080)), \
    std::make_pair(cv::Size(1920, 1080), cv::Size(1280, 1080)), \
    std::make_pair(cv::Size(1280,  720), cv::Size(1920,  720)), \
    std::make_pair(cv::Size(1280,  720), cv::Size( 640,  720)), \
    std::make_pair(cv::Size( 640,  480), cv::Size(1280,  480)), \
    std::make_pair(cv::Size( 640,  480), cv::Size( 320,  480)), \
    std::make_pair(cv::Size( 320,  200), cv::Size( 640,  200)), \
    std::make_pair(cv::Size( 320,  200), cv::Size( 113,  200)), \
    std::make_pair(cv::Size( 113,   71), cv::Size( 320,   71))

#define TEST_RESIZE_VERT \
    std::make_pair(cv::Size(3840, 2160), cv::Size(3840, 1080)), \
    std::make_pair(cv::Size(1920, 1080), cv::Size(1920, 2160)), \
    std::make_pair(cv::Size(1920, 1080), cv::Size(1920,  720)), \
    std::make_pair(cv::Size(1280,  720), cv::Size(1280, 1080)), \
    std::make_pair(cv::Size(1280,  720), cv::Size(1280,  480)), \
    std::make_pair(cv::Size( 640,  480), cv::Size( 640,  720)), \
    std::make_pair(cv::Size( 640,  480), cv::Size( 640,  200)), \
    std::make_pair(cv::Size( 320,  200), cv::Size( 320,  480)), \
    std::make_pair(cv::Size( 320,  200), cv::Size( 320,   71)), \
    std::make_pair(cv::Size( 113,   71), cv::Size( 113,  200))

#define TEST_RESIZE_COPY \
    std::make_pair(cv::Size(3840, 2160), cv::Size(3840, 2160)), \
    std::make_pair(cv::Size(1920, 1080), cv::Size(1920, 1080)), \
    std::make_pair(cv::Size(1280,  720), cv::Size(1280,  720)), \
    std::make_pair(cv::Size( 640,  480), cv::Size( 640,  480)), \
    std::make_pair(cv::Size( 320,  200), cv::Size( 320,  200)), \
    std::make_pair(cv::Size( 113,   71), cv::Size( 113,   71))

#define TEST_RESIZE_SPECIAL \
    std::make_pair(cv::Size(300, 300), cv::Size(300, 199)), \
    std::make_pair(cv::Size(300, 300), cv::Size(199, 300)), \
    std::make_pair(cv::Size(300, 300), cv::Size(199, 199)), \
    std::make_pair(cv::Size(199, 199), cv::Size(300, 300)), \
    std::make_pair(cv::Size(199, 300), cv::Size(300, 300)), \
    std::make_pair(cv::Size(300, 199), cv::Size(300, 300))

#define TEST_RESIZE_PAIRS \
    TEST_RESIZE_DOWN, \
    TEST_RESIZE_UP, \
    TEST_RESIZE_HORZ, \
    TEST_RESIZE_VERT, \
    TEST_RESIZE_COPY, \
    TEST_RESIZE_SPECIAL

using namespace testing;

INSTANTIATE_TEST_CASE_P(ResizeTestFluid_U8, ResizeTestGAPI,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(TEST_RESIZE_PAIRS),
                                Values(1))); // error not more than 1 unit

INSTANTIATE_TEST_CASE_P(ResizeTestFluid_F32, ResizeTestGAPI,
                        Combine(Values(CV_32FC1, CV_32FC3),
                                Values(cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(TEST_RESIZE_PAIRS),
                                Values(0.015))); // accuracy like ~1.5%

INSTANTIATE_TEST_CASE_P(SplitTestFluid, SplitTestGAPI,
                        Combine(Values(2, 3, 4),
                                Values(CV_8U, CV_32F),
                                Values(TEST_SIZES)));

INSTANTIATE_TEST_CASE_P(MergeTestFluid, MergeTestGAPI,
                        Combine(Values(2, 3, 4),
                                Values(CV_8U, CV_32F),
                                Values(TEST_SIZES)));

//----------------------------------------------------------------------

INSTANTIATE_TEST_CASE_P(ResizeTestFluid_U8, ResizeTestIE,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(TEST_RESIZE_PAIRS),
                                Values(1))); // error not more than 1 unit

INSTANTIATE_TEST_CASE_P(ResizeTestFluid_F32, ResizeTestIE,
                        Combine(Values(CV_32FC1, CV_32FC3),
                                Values(cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(TEST_RESIZE_PAIRS),
                                Values(0.05))); // error within 0.05 units

INSTANTIATE_TEST_CASE_P(SplitTestFluid, SplitTestIE,
                        Combine(Values(CV_8UC2, CV_8UC3, CV_8UC4,
                                       CV_32FC2, CV_32FC3, CV_32FC4),
                                Values(TEST_SIZES)));

INSTANTIATE_TEST_CASE_P(MergeTestFluid, MergeTestIE,
                        Combine(Values(CV_8UC2, CV_8UC3, CV_8UC4,
                                       CV_32FC2, CV_32FC3, CV_32FC4),
                                Values(TEST_SIZES)));

//------------------------------------------------------------------------------

namespace IE = InferenceEngine;

static const auto FRAME_SIZES =
   Values(std::make_pair(cv::Size(1920,1080),
                         cv::Size(1024,1024)), // person-vehicle-bike-detection-crossroad-0078
          std::make_pair(cv::Size(1024, 768),
                         cv::Size( 992, 544)), // person-detection-retail-0001
          std::make_pair(cv::Size(1280, 720),
                         cv::Size( 896, 512)), // road-segmentation-adas-0001
          std::make_pair(cv::Size(3840, 2160),
                         cv::Size(2048, 1024)), // semantic-segmentation-adas-0001
          std::make_pair(cv::Size(1270, 720),
                         cv::Size(2048, 1024)), // semantic-segmentation-adas-0001 (UPSCALE)
          std::make_pair(cv::Size( 640, 480),
                         cv::Size( 544, 320)));  // 320 - face-person-detection-retail-0002,
                                                 // 320 - person-detection-retail-10013
                                                 // 300 - face-detection-retail-0004

static const auto PATCH_SIZES =
    Values(std::make_pair(cv::Size(200,400),
                          cv::Size(128,384)),  // person-reidentification-retail-0076
           std::make_pair(cv::Size( 96,256),
                          cv::Size(128,384)),  // person-reidentification-retail-0076 (UPSCALE)
           std::make_pair(cv::Size(340,340),
                          cv::Size(320,256)),  // vehicle-license-plate-detection-barrier-0007
           std::make_pair(cv::Size(256,256),
                          cv::Size( 72,72)),   // vehicle-attributes-recognition-barrier-0039
           std::make_pair(cv::Size(96,96),
                          cv::Size(64,64)),    // 60 - head-pose-estimation-adas-0001
                                               // 62 - age-gender-recognition-retail-0013
                                               // 64 - emotions-recognition-retail-0003
           std::make_pair(cv::Size(128,48),
                          cv::Size( 94,24)),   // license-plate-recognition-barrier-0001
           std::make_pair(cv::Size(120,200),
                          cv::Size(80, 160))); // 80 - person-attributes-recognition-crossroad-0031
                                               // 64 - person-reidentification-retail-0079

INSTANTIATE_TEST_CASE_P(ReorderResize_Frame, PreprocTest,
                        Combine(Values(IE::Precision::U8, IE::Precision::FP32),
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR), // AREA is not there yet
                                Values(IE::Layout::NHWC),
                                Values(IE::Layout::NCHW),
                                Values(1, 3),
                                FRAME_SIZES));

INSTANTIATE_TEST_CASE_P(Scale3ch_Frame, PreprocTest,
                        Combine(Values(IE::Precision::U8, IE::Precision::FP32),
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR), // AREA is not there yet
                                Values(IE::Layout::NHWC),
                                Values(IE::Layout::NHWC),
                                Values(3),
                                FRAME_SIZES));

INSTANTIATE_TEST_CASE_P(ReorderResize_Patch, PreprocTest,
                        Combine(Values(IE::Precision::U8, IE::Precision::FP32),
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR), // AREA is not there yet
                                Values(IE::Layout::NHWC),
                                Values(IE::Layout::NCHW, IE::Layout::NCHW),
                                Values(1, 3),
                                PATCH_SIZES));

INSTANTIATE_TEST_CASE_P(Everything, PreprocTest,
                        Combine(Values(IE::Precision::U8, IE::Precision::FP32),
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR, IE::ResizeAlgorithm::RESIZE_AREA),
                                Values(IE::Layout::NHWC, IE::Layout::NCHW),
                                Values(IE::Layout::NHWC, IE::Layout::NCHW),
                                Values(1, 2, 3, 4),
                                Values(std::make_pair(cv::Size(1920, 1080), cv::Size(1024,1024)),
                                       std::make_pair(cv::Size(1280, 720), cv::Size(544,320)),
                                       std::make_pair(cv::Size(640, 480), cv::Size(896, 512)),
                                       std::make_pair(cv::Size(200, 400), cv::Size(128, 384)),
                                       std::make_pair(cv::Size(256, 256), cv::Size(72, 72)),
                                       std::make_pair(cv::Size(96, 256), cv::Size(128, 384)))));

}
