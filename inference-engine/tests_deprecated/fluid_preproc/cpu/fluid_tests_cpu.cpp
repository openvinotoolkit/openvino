// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fluid_tests.hpp"

#include <opencv2/opencv.hpp>

#include <gtest/gtest.h>

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

#define TEST_SIZES_PREPROC \
    std::make_pair(cv::Size(1920, 1080), cv::Size(1024, 1024)), \
    std::make_pair(cv::Size(1280,  720), cv::Size( 544,  320)), \
    std::make_pair(cv::Size( 640,  480), cv::Size( 896,  512)), \
    std::make_pair(cv::Size( 200,  400), cv::Size( 128,  384)), \
    std::make_pair(cv::Size( 256,  256), cv::Size(  72,   72)), \
    std::make_pair(cv::Size(  96,  256), cv::Size( 128,  384))

using namespace testing;
#if defined(__arm__) || defined(__aarch64__)
INSTANTIATE_TEST_SUITE_P(ResizeTestFluid_U8, ResizeTestGAPI,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(TEST_RESIZE_PAIRS),
                                Values(4))); // error not more than 4 unit

INSTANTIATE_TEST_SUITE_P(ResizeRGB8UTestFluid_U8, ResizeRGB8UTestGAPI,
                        Combine(Values(CV_8UC3, CV_8UC4),
                                Values(cv::INTER_LINEAR),
                                Values(TEST_RESIZE_PAIRS),
                                Values(4))); // error not more than 4 unit
#else
INSTANTIATE_TEST_SUITE_P(ResizeTestFluid_U8, ResizeTestGAPI,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(TEST_RESIZE_PAIRS),
                                Values(1))); // error not more than 1 unit

INSTANTIATE_TEST_SUITE_P(ResizeRGB8UTestFluid_U8, ResizeRGB8UTestGAPI,
                        Combine(Values(CV_8UC3, CV_8UC4),
                                Values(cv::INTER_LINEAR),
                                Values(TEST_RESIZE_PAIRS),
                                Values(1))); // error not more than 1 unit
#endif

INSTANTIATE_TEST_SUITE_P(ResizeTestFluid_F32, ResizeTestGAPI,
                        Combine(Values(CV_32FC1, CV_32FC3),
                                Values(cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(TEST_RESIZE_PAIRS),
                                Values(0.015))); // accuracy like ~1.5%


INSTANTIATE_TEST_SUITE_P(SplitTestFluid, SplitTestGAPI,
                        Combine(Values(2, 3, 4),
                                Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_16F, CV_32F, CV_32S),
                                Values(TEST_SIZES),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(ChanToPlaneTestFluid, ChanToPlaneTestGAPI,
                        Combine(Values(1, 3),
                                Values(CV_8U, CV_32F),
                                Values(TEST_SIZES),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(MergeTestFluid, MergeTestGAPI,
                        Combine(Values(2, 3, 4),
                                Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_16F, CV_32F, CV_32S),
                                Values(TEST_SIZES),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(NV12toRGBTestFluid, NV12toRGBTestGAPI,
                        Combine(Values(cv::Size(3840, 2160),
                                       cv::Size(1920, 1080),
                                       cv::Size(1280,  720),
                                       cv::Size(1280,  960),
                                       cv::Size( 960,  720),
                                       cv::Size( 640,  480),
                                       cv::Size( 300,  300),
                                       cv::Size( 320,  200)),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(I420toRGBTestFluid, I420toRGBTestGAPI,
                        Combine(Values(cv::Size(3840, 2160),
                                       cv::Size(1920, 1080),
                                       cv::Size(1280,  720),
                                       cv::Size(1280,  960),
                                       cv::Size( 960,  720),
                                       cv::Size( 640,  480),
                                       cv::Size( 300,  300),
                                       cv::Size( 320,  200)),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(ConvertDepthFluid, ConvertDepthTestGAPI,
                        Combine(Values(CV_16U, CV_32F, CV_8U),
                                Values(CV_32F, CV_16U, CV_8U),
                                Values(cv::Size(3840, 2160),
                                       cv::Size(1920, 1080),
                                       cv::Size(1280,  720),
                                       cv::Size(1280,  960),
                                       cv::Size( 960,  720),
                                       cv::Size( 640,  480),
                                       cv::Size( 300,  300),
                                       cv::Size( 320,  200)),
                                Values(1)));

INSTANTIATE_TEST_SUITE_P(DivCFluid, DivCTestGAPI,
                        Combine(Values(CV_32F),
                                Values(1),      //channels
                                Values(TEST_SIZES),
                                Values(cv::Scalar{0.229}),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(SubCFluid, SubCTestGAPI,
                        Combine(Values(CV_32F),
                                Values(1),      //channels
                                Values(TEST_SIZES),
                                Values(cv::Scalar{0.229}),
                                Values(0.00001)));

INSTANTIATE_TEST_SUITE_P(ResizeRoiTestFluid, ResizeRoiTestGAPI,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::INTER_LINEAR),
                                Values(std::make_pair(cv::Size(24, 24), cv::Size(12, 12))),
                                Values(cv::Rect{0, 0, 12, 3},
                                       cv::Rect{0, 3, 12, 3},
                                       cv::Rect{0, 6, 12, 3},
                                       cv::Rect{0, 9, 12, 3}),
                                Values(1))); // error not more than 1 unit

INSTANTIATE_TEST_SUITE_P(ResizeRGB8URoiTestFluid, ResizeRGB8URoiTestGAPI,
                        Combine(Values(CV_8UC3, CV_8UC4),
                                Values(cv::INTER_LINEAR),
                                Values(std::make_pair(cv::Size(24, 24), cv::Size(12, 12))),
                                Values(cv::Rect{0, 0, 12, 3},
                                       cv::Rect{0, 3, 12, 3},
                                       cv::Rect{0, 6, 12, 3},
                                       cv::Rect{0, 9, 12, 3}),
                                Values(1))); // error not more than 1 unit

//----------------------------------------------------------------------

#if defined(__arm__) || defined(__aarch64__)
INSTANTIATE_TEST_SUITE_P(ResizeTestFluid_U8, ResizeTestIE,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(TEST_RESIZE_PAIRS),
                                Values(4))); // error not more than 4 unit
#else
INSTANTIATE_TEST_SUITE_P(ResizeTestFluid_U8, ResizeTestIE,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(TEST_RESIZE_PAIRS),
                                Values(1))); // error not more than 1 unit
#endif

INSTANTIATE_TEST_SUITE_P(ResizeTestFluid_F32, ResizeTestIE,
                        Combine(Values(CV_32FC1, CV_32FC3),
                                Values(cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(TEST_RESIZE_PAIRS),
                                Values(0.05))); // error within 0.05 units

INSTANTIATE_TEST_SUITE_P(SplitTestFluid, SplitTestIE,
                        Combine(Values(CV_8UC2, CV_8UC3, CV_8UC4,
                                       CV_32FC2, CV_32FC3, CV_32FC4),
                                Values(TEST_SIZES),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(MergeTestFluid, MergeTestIE,
                        Combine(Values(CV_8UC2, CV_8UC3, CV_8UC4,
                                       CV_32FC2, CV_32FC3, CV_32FC4),
                                Values(TEST_SIZES),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(ColorConvertFluid_3ch, ColorConvertTestIE,
                        Combine(Values(CV_8U, CV_32F),
                                Values(InferenceEngine::ColorFormat::RGB),
                                Values(InferenceEngine::NHWC, InferenceEngine::NCHW),
                                Values(InferenceEngine::NHWC, InferenceEngine::NCHW),
                                Values(TEST_SIZES),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(ColorConvertFluid_4ch, ColorConvertTestIE,
                        Combine(Values(CV_8U, CV_32F),
                                Values(InferenceEngine::ColorFormat::RGBX,
                                       InferenceEngine::ColorFormat::BGRX),
                                Values(InferenceEngine::NHWC),
                                Values(InferenceEngine::NHWC, InferenceEngine::NCHW),
                                Values(TEST_SIZES),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(ColorConvertYUV420Fluid, ColorConvertYUV420TestIE,
                        Combine(Values(InferenceEngine::NV12, InferenceEngine::I420),
                                Values(InferenceEngine::NHWC, InferenceEngine::NCHW),
                                Values(cv::Size(3840, 2160),
                                       cv::Size(1920, 1080),
                                       cv::Size(1280,  720),
                                       cv::Size(1280,  960),
                                       cv::Size( 960,  720),
                                       cv::Size( 640,  480),
                                       cv::Size( 320,  200),
                                       cv::Size( 300,  300),
                                       cv::Size( 150,  150)),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(Reorder_HWC2CHW, ColorConvertTestIE,
                        Combine(Values(CV_8U, CV_32F, CV_16S, CV_16F),
                                Values(InferenceEngine::ColorFormat::BGR),
                                Values(InferenceEngine::NHWC),
                                Values(InferenceEngine::NCHW),
                                Values(TEST_SIZES),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(Reorder_CHW2HWC, ColorConvertTestIE,
                        Combine(Values(CV_8U, CV_32F, CV_16S, CV_16F),
                                Values(InferenceEngine::ColorFormat::BGR),
                                Values(InferenceEngine::NCHW),
                                Values(InferenceEngine::NHWC),
                                Values(TEST_SIZES),
                                Values(0)));

INSTANTIATE_TEST_SUITE_P(MeanValueGAPI32F, MeanValueGAPI,
                        Combine(Values(TEST_SIZES),
                                Values(0.00001)));

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

static const auto U8toU8 = std::make_pair(IE::Precision::U8,IE::Precision::U8);

static const auto PRECISIONS = Values(U8toU8, std::make_pair(IE::Precision::FP32,IE::Precision::FP32));

INSTANTIATE_TEST_SUITE_P(ReorderResize_Frame, PreprocTest,
                        Combine(PRECISIONS,
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR), // AREA is not there yet
                                Values(IE::ColorFormat::RAW),
                                Values(IE::Layout::NHWC),
                                Values(IE::Layout::NCHW),
                                Values(std::make_pair(1, 1), std::make_pair(3, 3)),
                                FRAME_SIZES));

INSTANTIATE_TEST_SUITE_P(Scale3ch_Frame, PreprocTest,
                        Combine(PRECISIONS,
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR), // AREA is not there yet
                                Values(IE::ColorFormat::RAW),
                                Values(IE::Layout::NHWC),
                                Values(IE::Layout::NHWC),
                                Values(std::make_pair(3, 3)),
                                FRAME_SIZES));

INSTANTIATE_TEST_SUITE_P(ReorderResize_Patch, PreprocTest,
                        Combine(PRECISIONS,
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR), // AREA is not there yet
                                Values(IE::ColorFormat::RAW),
                                Values(IE::Layout::NHWC),
                                Values(IE::Layout::NCHW, IE::Layout::NCHW),
                                Values(std::make_pair(1, 1), std::make_pair(3, 3)),
                                PATCH_SIZES));

INSTANTIATE_TEST_SUITE_P(Everything_Resize, PreprocTest,
                        Combine(PRECISIONS,
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR, IE::ResizeAlgorithm::RESIZE_AREA),
                                Values(IE::ColorFormat::RAW),
                                Values(IE::Layout::NHWC, IE::Layout::NCHW),
                                Values(IE::Layout::NHWC, IE::Layout::NCHW),
                                Values(std::make_pair(1, 1),
                                       std::make_pair(2, 2),
                                       std::make_pair(3, 3),
                                       std::make_pair(4, 4)),
                                Values(TEST_SIZES_PREPROC)));

INSTANTIATE_TEST_SUITE_P(ColorFormats_3ch, PreprocTest,
                        Combine(PRECISIONS,
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR, IE::ResizeAlgorithm::RESIZE_AREA),
                                Values(IE::ColorFormat::RGB),
                                Values(IE::Layout::NHWC, IE::Layout::NCHW),
                                Values(IE::Layout::NHWC, IE::Layout::NCHW),
                                Values(std::make_pair(3, 3)),
                                Values(TEST_SIZES_PREPROC)));

INSTANTIATE_TEST_SUITE_P(ColorFormats_4ch, PreprocTest,
                        Combine(PRECISIONS,
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR, IE::ResizeAlgorithm::RESIZE_AREA),
                                Values(IE::ColorFormat::BGRX, IE::ColorFormat::RGBX),
                                Values(IE::Layout::NHWC),
                                Values(IE::Layout::NHWC, IE::Layout::NCHW),
                                Values(std::make_pair(4, 3)),
                                Values(TEST_SIZES_PREPROC)));

INSTANTIATE_TEST_SUITE_P(ColorFormat_NV12, PreprocTest,
                        Combine(Values(U8toU8),
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR, IE::ResizeAlgorithm::RESIZE_AREA),
                                Values(IE::ColorFormat::NV12),
                                Values(IE::Layout::NCHW),
                                Values(IE::Layout::NHWC, IE::Layout::NCHW),
                                Values(std::make_pair(1, 3)),
                                Values(TEST_SIZES_PREPROC)));


INSTANTIATE_TEST_SUITE_P(PlainPrecisionConversions, PreprocTest,
                        Combine(Values(std::make_pair(IE::Precision::U16,IE::Precision::FP32),
                                       std::make_pair(IE::Precision::FP32,IE::Precision::U16)
                                ),
                                Values(IE::ResizeAlgorithm::NO_RESIZE),
                                Values(IE::ColorFormat::RAW),
                                Values(IE::Layout::NHWC),
                                Values(IE::Layout::NHWC),
                                Values(std::make_pair(1, 1)),
                                Values(std::make_pair(cv::Size(640,480), cv::Size(640,480)))));


INSTANTIATE_TEST_SUITE_P(PrecisionConversionsPipelines, PreprocTest,
                        Combine(Values(std::make_pair(IE::Precision::U16, IE::Precision::FP32),
                                       std::make_pair(IE::Precision::FP32,IE::Precision::U8),
                                       std::make_pair(IE::Precision::U8,  IE::Precision::FP32)
                                ),
                                Values(IE::ResizeAlgorithm::RESIZE_BILINEAR),
                                Values(IE::ColorFormat::RAW),
                                Values(IE::Layout::NHWC, IE::Layout::NCHW),
                                Values(IE::Layout::NHWC, IE::Layout::NCHW),
                                Values(std::make_pair(1, 1), std::make_pair(3, 3)),
                                Values(TEST_SIZES_PREPROC)));
