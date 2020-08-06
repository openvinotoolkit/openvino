// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fluid_tests.hpp"

#include "blob_factory.hpp"
#include "blob_transform.hpp"
#include "ie_preprocess.hpp"
#include "ie_preprocess_data.hpp"
#include "ie_compound_blob.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/imgproc.hpp>

#include <cstdarg>
#include <cstdio>
#include <ctime>

#include <chrono>

#include <map>

#include <fluid_test_computations.hpp>

// Can be set externally (via CMake) if built with -DGAPI_TEST_PERF=ON
#ifndef PERF_TEST
#define PERF_TEST 0 // 1=test performance, 0=don't
#endif

namespace {
#if PERF_TEST
// performance test: iterate function, measure and print milliseconds per call
template<typename F> void test_ms(F func, int iter, const char format[], ...)
{
    using std::chrono::high_resolution_clock;

    std::vector<high_resolution_clock::duration> samples(iter); samples.clear();
    if (0 == iter)
        return;

    for (int i=0; i < iter; i++)
    {
        auto start = high_resolution_clock::now();
        func(); // iterate calls
        samples.push_back(high_resolution_clock::now() - start);
    }

    std::sort(samples.begin(), samples.end());

    auto median = samples[samples.size() / 2];

    double median_ms = std::chrono::duration_cast<std::chrono::microseconds>(median).count() * 0.001; // convert to milliseconds

    printf("Performance(ms): %lg ", median_ms);

    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);

    printf("\n");
}

cv::String interpToString(int interp)
{
    switch(interp)
    {
    case cv::INTER_AREA   : return "INTER_AREA";
    case cv::INTER_LINEAR : return "INTER_LINEAR";
    case cv::INTER_NEAREST: return "INTER_NEAREST";
    }
    CV_Assert(!"ERROR: unsupported interpolation!");
    return nullptr;
}

cv::String depthToString(int depth)
{
    switch(depth)
    {
    case CV_8U  : return "CV_8U";
    case CV_32F : return "CV_32F";
    }
    CV_Assert(!"ERROR: unsupported depth!");
    return nullptr;
}

cv::String typeToString(int type)
{
    switch(type)
    {
    case CV_8UC1  : return "CV_8UC1";
    case CV_8UC2  : return "CV_8UC2";
    case CV_8UC3  : return "CV_8UC3";
    case CV_8UC4  : return "CV_8UC4";
    case CV_32FC1 : return "CV_32FC1";
    case CV_32FC2 : return "CV_32FC2";
    case CV_32FC3 : return "CV_32FC3";
    case CV_32FC4 : return "CV_32FC4";
    }
    CV_Assert(!"ERROR: unsupported type!");
    return nullptr;
}

cv::String colorFormatToString(InferenceEngine::ColorFormat f) {
    using namespace InferenceEngine;
    switch (f)
    {
        case ColorFormat::RAW: return "RAW";
        case ColorFormat::RGB: return "RGB";
        case ColorFormat::BGR: return "BGR";
        case ColorFormat::RGBX: return "RGBX";
        case ColorFormat::BGRX: return "BGRX";
        case ColorFormat::NV12: return "NV12";
        default: THROW_IE_EXCEPTION << "Unrecognized color format";
    }
}

cv::String layoutToString(InferenceEngine::Layout l) {
    using namespace InferenceEngine;
    switch (l) {
    case Layout::NCHW: return "NCHW";
    case Layout::NHWC: return "NHWC";
    default: return "?";
    }
}
#endif  // PERF_TEST

test::Mat to_test(cv::Mat& mat) { return {mat.rows, mat.cols, mat.type(), mat.data, mat.step}; }
std::vector<test::Mat> to_test(std::vector<cv::Mat>& mats)
{
    std::vector<test::Mat> test_mats(mats.size());
    for (int i = 0; i < mats.size(); i++) {
        test_mats[i] = to_test(mats[i]);
    }
    return test_mats;
}

test::Rect to_test(cv::Rect& rect) { return {rect.x, rect.y, rect.width, rect.height}; }

cv::ColorConversionCodes toCvtColorCode(InferenceEngine::ColorFormat in,
                                     InferenceEngine::ColorFormat out) {
    using namespace InferenceEngine;
    static const std::map<std::pair<ColorFormat, ColorFormat>, cv::ColorConversionCodes> types = {
        {{ColorFormat::RGBX, ColorFormat::BGRX}, cv::COLOR_RGBA2BGRA},
        {{ColorFormat::RGBX, ColorFormat::BGR}, cv::COLOR_RGBA2BGR},
        {{ColorFormat::RGBX, ColorFormat::RGB}, cv::COLOR_RGBA2RGB},
        {{ColorFormat::BGRX, ColorFormat::RGBX}, cv::COLOR_BGRA2RGBA},
        {{ColorFormat::BGRX, ColorFormat::BGR}, cv::COLOR_BGRA2BGR},
        {{ColorFormat::BGRX, ColorFormat::RGB}, cv::COLOR_BGRA2RGB},
        {{ColorFormat::RGB, ColorFormat::RGBX}, cv::COLOR_RGB2RGBA},
        {{ColorFormat::RGB, ColorFormat::BGRX}, cv::COLOR_RGB2BGRA},
        {{ColorFormat::RGB, ColorFormat::BGR}, cv::COLOR_RGB2BGR},
        {{ColorFormat::BGR, ColorFormat::RGBX}, cv::COLOR_BGR2RGBA},
        {{ColorFormat::BGR, ColorFormat::BGRX}, cv::COLOR_BGR2BGRA},
        {{ColorFormat::BGR, ColorFormat::RGB}, cv::COLOR_BGR2RGB},
        {{ColorFormat::NV12, ColorFormat::BGR}, cv::COLOR_YUV2BGR_NV12},
        {{ColorFormat::NV12, ColorFormat::RGB}, cv::COLOR_YUV2RGB_NV12}
    };
    return types.at(std::make_pair(in, out));
}

cv::ColorConversionCodes toCvtColorCode(InferenceEngine::ColorFormat fmt) {
    using namespace InferenceEngine;
    // Note: OpenCV matrices are always in BGR format by default
    return toCvtColorCode(ColorFormat::BGR, fmt);
}

int numChannels(InferenceEngine::ColorFormat fmt) {
    using namespace InferenceEngine;
    switch (fmt) {
        // case ColorFormat::RAW: return 0;  // any number of channels apply
        case ColorFormat::RGB: return 3;
        case ColorFormat::BGR: return 3;
        case ColorFormat::RGBX: return 4;
        case ColorFormat::BGRX: return 4;
        default: THROW_IE_EXCEPTION << "Unrecognized color format";
    }
}

// FIXME: Copy-paste from cropRoi tests
template <InferenceEngine::Precision::ePrecision PRC>
InferenceEngine::Blob::Ptr img2Blob(cv::Mat &img, InferenceEngine::Layout layout) {
    using namespace InferenceEngine;
    using data_t = typename PrecisionTrait<PRC>::value_type;

    const size_t channels = img.channels();
    const size_t height = img.size().height;
    const size_t width = img.size().width;

    CV_Assert(cv::DataType<data_t>::depth == img.depth());

    SizeVector dims = {1, channels, height, width};
    Blob::Ptr resultBlob = make_shared_blob<data_t>(TensorDesc(PRC, dims, layout));;
    resultBlob->allocate();

    data_t* blobData = resultBlob->buffer().as<data_t*>();

    switch (layout) {
        case Layout::NCHW: {
            for (size_t c = 0; c < channels; c++) {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        blobData[c * width * height + h * width + w] = img.ptr<data_t>(h,w)[c];
                    }
                }
            }
        }
        break;
        case Layout::NHWC: {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    for (size_t c = 0; c < channels; c++) {
                        blobData[h * width * channels + w * channels + c] = img.ptr<data_t>(h,w)[c];
                    }
                }
            }
        }
        break;
        default:
            THROW_IE_EXCEPTION << "Inconsistent input layout for image processing: " << layout;
    }
    return resultBlob;
}

template <InferenceEngine::Precision::ePrecision PRC>
void Blob2Img(const InferenceEngine::Blob::Ptr& blobP, cv::Mat& img, InferenceEngine::Layout layout) {
    using namespace InferenceEngine;
    using data_t = typename PrecisionTrait<PRC>::value_type;

    const size_t channels = img.channels();
    const size_t height = img.size().height;
    const size_t width = img.size().width;

    CV_Assert(cv::DataType<data_t>::depth == img.depth());

    data_t* blobData = blobP->buffer().as<data_t*>();

    switch (layout) {
        case Layout::NCHW: {
            for (size_t c = 0; c < channels; c++) {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        img.ptr<data_t>(h,w)[c] = blobData[c * width * height + h * width + w];
                    }
                }
            }
        }
        break;
        case Layout::NHWC: {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    for (size_t c = 0; c < channels; c++) {
                        img.ptr<data_t>(h,w)[c] = blobData[h * width * channels + w * channels + c];
                    }
                }
            }
        }
        break;
        default:
            THROW_IE_EXCEPTION << "Inconsistent input layout for image processing: " << layout;
    }
}
} // anonymous namespace

TEST_P(ResizeTestGAPI, AccuracyTest)
{
    int type = 0, interp = 0;
    cv::Size sz_in, sz_out;
    double tolerance = 0.0;
    std::pair<cv::Size, cv::Size> sizes;
    std::tie(type, interp, sizes, tolerance) = GetParam();
    std::tie(sz_in, sz_out) = sizes;

    cv::Mat in_mat1 (sz_in, type );
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    cv::Mat out_mat_ocv(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    FluidResizeComputation rc(to_test(in_mat1), to_test(out_mat), interp);
    rc.warmUp();

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ rc.apply(); },
            100, "Resize GAPI %s %s %dx%d -> %dx%d",
            interpToString(interp).c_str(), typeToString(type).c_str(),
            sz_in.width, sz_in.height, sz_out.width, sz_out.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::resize(in_mat1, out_mat_ocv, sz_out, 0, 0, interp);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat, out_mat_ocv, cv::NORM_INF), tolerance);
    }
}

TEST_P(ResizeRGB8UTestGAPI, AccuracyTest)
{
    int type = 0, interp = 0;
    cv::Size sz_in, sz_out;
    double tolerance = 0.0;
    std::pair<cv::Size, cv::Size> sizes;
    std::tie(type, interp, sizes, tolerance) = GetParam();
    std::tie(sz_in, sz_out) = sizes;

    cv::Mat in_mat1 (sz_in, type );
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    cv::Mat out_mat_ocv(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    FluidResizeRGB8UComputation rc(to_test(in_mat1), to_test(out_mat), interp);
    rc.warmUp();

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ rc.apply(); },
            100, "Resize GAPI %s %s %dx%d -> %dx%d",
            interpToString(interp).c_str(), typeToString(type).c_str(),
            sz_in.width, sz_in.height, sz_out.width, sz_out.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::resize(in_mat1, out_mat_ocv, sz_out, 0, 0, interp);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat, out_mat_ocv, cv::NORM_INF), tolerance);
    }
}

TEST_P(ResizeRoiTestGAPI, AccuracyTest)
{
    int type = 0, interp = 0;
    cv::Size sz_in, sz_out;
    cv::Rect roi;
    double tolerance = 0.0;
    std::pair<cv::Size, cv::Size> sizes;
    std::tie(type, interp, sizes, roi, tolerance) = GetParam();
    std::tie(sz_in, sz_out) = sizes;

    cv::Mat in_mat1 (sz_in, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    cv::Mat out_mat_ocv(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    FluidResizeComputation rc(to_test(in_mat1), to_test(out_mat), interp);
    rc.warmUp(to_test(roi));

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ rc.apply(); },
            100, "Resize GAPI %s %s %dx%d -> %dx%d",
            interpToString(interp).c_str(), typeToString(type).c_str(),
            sz_in.width, sz_in.height, sz_out.width, sz_out.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::resize(in_mat1, out_mat_ocv, sz_out, 0, 0, interp);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat(roi), out_mat_ocv(roi), cv::NORM_INF), tolerance);
    }
}

TEST_P(ResizeRGB8URoiTestGAPI, AccuracyTest)
{
    int type = 0, interp = 0;
    cv::Size sz_in, sz_out;
    cv::Rect roi;
    double tolerance = 0.0;
    std::pair<cv::Size, cv::Size> sizes;
    std::tie(type, interp, sizes, roi, tolerance) = GetParam();
    std::tie(sz_in, sz_out) = sizes;

    cv::Mat in_mat1 (sz_in, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    cv::Mat out_mat_ocv(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    FluidResizeRGB8UComputation rc(to_test(in_mat1), to_test(out_mat), interp);
    rc.warmUp(to_test(roi));

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ rc.apply(); },
            100, "Resize GAPI %s %s %dx%d -> %dx%d",
            interpToString(interp).c_str(), typeToString(type).c_str(),
            sz_in.width, sz_in.height, sz_out.width, sz_out.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::resize(in_mat1, out_mat_ocv, sz_out, 0, 0, interp);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat(roi), out_mat_ocv(roi), cv::NORM_INF), tolerance);
    }
}

TEST_P(SplitTestGAPI, AccuracyTest)
{
    const auto params = GetParam();
    int planes  = std::get<0>(params);
    int depth   = std::get<1>(params);
    cv::Size sz = std::get<2>(params);
    double tolerance = std::get<3>(params);

    int srcType = CV_MAKE_TYPE(depth, planes);
    int dstType = CV_MAKE_TYPE(depth, 1);

    cv::Mat in_mat(sz, srcType);
    cv::randn(in_mat, cv::Scalar::all(127), cv::Scalar::all(40.f));

    std::vector<cv::Mat> out_mats_gapi(planes, cv::Mat::zeros(sz, dstType));
    std::vector<cv::Mat> out_mats_ocv (planes, cv::Mat::zeros(sz, dstType));

    // G-API code //////////////////////////////////////////////////////////////
    FluidSplitComputation sc(to_test(in_mat), to_test(out_mats_gapi));
    sc.warmUp();

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ sc.apply(); },
        400, "Split GAPI %s %dx%d", typeToString(srcType).c_str(), sz.width, sz.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::split(in_mat, out_mats_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        for (int p = 0; p < planes; p++) {
            EXPECT_LE(cv::norm(out_mats_ocv[p], out_mats_gapi[p], cv::NORM_INF), tolerance);
        }
    }
}

TEST_P(ChanToPlaneTestGAPI, AccuracyTest)
{
    const auto params = GetParam();
    int planes  = std::get<0>(params);
    int depth   = std::get<1>(params);
    cv::Size sz = std::get<2>(params);
    double tolerance = std::get<3>(params);

    int inType  = CV_MAKE_TYPE(depth, planes);
    int outType = CV_MAKE_TYPE(depth, 1);

    cv::Mat in_mat(sz, inType);
    cv::randn(in_mat, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::Mat out_mat_gapi(sz, outType);
    std::vector<cv::Mat> out_mats_ocv;

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::split(in_mat, out_mats_ocv);
    }

    for(int i = 0; i < planes; ++i){
        // G-API code //////////////////////////////////////////////////////////////
        FluidChanToPlaneComputation sc(to_test(in_mat), to_test(out_mat_gapi), i);
        sc.warmUp();

        #if PERF_TEST
            // run just for a single plane
            if(i == 0){
                // iterate testing, and print performance
                test_ms([&](){ sc.apply(); },
                        400, "ChanToPlane GAPI %s %dx%d", typeToString(inType).c_str(), sz.width, sz.height);
            }
        #endif

        // Comparison //////////////////////////////////////////////////////////////
        {
            EXPECT_LE(cv::norm(out_mats_ocv[i], out_mat_gapi, cv::NORM_INF), tolerance);
        }
    }
}

TEST_P(MergeTestGAPI, AccuracyTest)
{
    const auto params = GetParam();
    int planes  = std::get<0>(params);
    int depth   = std::get<1>(params);
    cv::Size sz = std::get<2>(params);
    double tolerance = std::get<3>(params);

    int srcType = CV_MAKE_TYPE(depth, 1);
    int dstType = CV_MAKE_TYPE(depth, planes);

    std::vector<cv::Mat> in_mats(planes, cv::Mat(sz, srcType));
    for (int p = 0; p < planes; p++) {
        cv::randn(in_mats[p], cv::Scalar::all(127), cv::Scalar::all(40.f));
    }

    cv::Mat out_mat_ocv  = cv::Mat::zeros(sz, dstType);
    cv::Mat out_mat_gapi = cv::Mat::zeros(sz, dstType);

    // G-API code //////////////////////////////////////////////////////////////
    FluidMergeComputation mc(to_test(in_mats), to_test(out_mat_gapi));
    mc.warmUp();

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ mc.apply(); },
        400, "Merge GAPI %s %dx%d", typeToString(dstType).c_str(), sz.width, sz.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::merge(in_mats, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat_ocv, out_mat_gapi, cv::NORM_INF), tolerance);
    }
}

TEST_P(NV12toRGBTestGAPI, AccuracyTest)
{
    const auto params = GetParam();
    cv::Size sz = std::get<0>(params);
    double tolerance = std::get<1>(params);

    cv::Mat in_mat_y(sz, CV_8UC1);
    cv::Mat in_mat_uv(cv::Size(sz.width / 2, sz.height / 2), CV_8UC2);
    cv::randn(in_mat_y, cv::Scalar::all(127), cv::Scalar::all(40.f));
    cv::randn(in_mat_uv, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::Mat out_mat_gapi(cv::Mat::zeros(sz, CV_8UC3));
    cv::Mat out_mat_ocv (cv::Mat::zeros(sz, CV_8UC3));

    // G-API code //////////////////////////////////////////////////////////////
    FluidNV12toRGBComputation cc(to_test(in_mat_y), to_test(in_mat_uv), to_test(out_mat_gapi));
    cc.warmUp();

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ cc.apply(); },
        400, "NV12toRGB GAPI %s %dx%d", typeToString(CV_8UC3).c_str(), sz.width, sz.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColorTwoPlane(in_mat_y,in_mat_uv,out_mat_ocv,cv::COLOR_YUV2RGB_NV12);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat_ocv, out_mat_gapi, cv::NORM_INF), tolerance);
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}


TEST_P(I420toRGBTestGAPI, AccuracyTest)
{
    const auto params = GetParam();
    cv::Size sz = std::get<0>(params);
    double tolerance = std::get<1>(params);

    cv::Mat in_mat_y(sz, CV_8UC1);
    cv::Mat in_mat_u(cv::Size(sz.width / 2, sz.height / 2), CV_8UC1);
    cv::Mat in_mat_v(cv::Size(sz.width / 2, sz.height / 2), CV_8UC1);
    cv::randn(in_mat_y, cv::Scalar::all(127), cv::Scalar::all(40.f));
    cv::randn(in_mat_u, cv::Scalar::all(127), cv::Scalar::all(40.f));
    cv::randn(in_mat_v, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::Mat out_mat_gapi(cv::Mat::zeros(sz, CV_8UC3));
    cv::Mat out_mat_ocv (cv::Mat::zeros(sz, CV_8UC3));

    // G-API code //////////////////////////////////////////////////////////////
    FluidI420toRGBComputation cc(to_test(in_mat_y), to_test(in_mat_u), to_test(in_mat_v), to_test(out_mat_gapi));
    cc.warmUp();

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ cc.apply(); },
        400, "I420toRGB GAPI %s %dx%d", typeToString(CV_8UC3).c_str(), sz.width, sz.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Mat in_mat_uv = cv::Mat::zeros(in_mat_u.size(), CV_8UC2);
        std::array<cv::Mat, 2> in_uv = {in_mat_u, in_mat_v};
        cv::merge(in_uv, in_mat_uv);
        //cvtColorTwoPlane supports NV12 only at the moment
        cv::cvtColorTwoPlane(in_mat_y,in_mat_uv,out_mat_ocv,cv::COLOR_YUV2RGB_NV12);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat_ocv, out_mat_gapi, cv::NORM_INF), tolerance);
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(U16toF32TestGAPI, AccuracyTest)
{
    const auto params = GetParam();
    cv::Size sz      = std::get<0>(params);
    double tolerance = std::get<1>(params);

    initMatrixRandU(CV_16UC1, sz, CV_32FC1);

    // G-API code //////////////////////////////////////////////////////////////
    FluidU16ToF32Computation cc(to_test(in_mat1), to_test(out_mat_gapi));
    cc.warmUp();

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ cc.apply(); },
        400, "U16ToF32 GAPI %s %dx%d", typeToString(CV_16UC1).c_str(), sz.width, sz.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        in_mat1.convertTo(out_mat_ocv, CV_32FC1);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat_ocv, out_mat_gapi, cv::NORM_INF), tolerance);
    }
}
//----------------------------------------------------------------------

TEST_P(ResizeTestIE, AccuracyTest)
{
    int type = 0, interp = 0;
    cv::Size sz_in, sz_out;
    double tolerance = 0.0;
    std::pair<cv::Size, cv::Size> sizes;
    std::tie(type, interp, sizes, tolerance) = GetParam();
    std::tie(sz_in, sz_out) = sizes;

    cv::Mat in_mat1(sz_in, type );
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    cv::Mat out_mat_ocv(sz_out, type);

    // Inference Engine code ///////////////////////////////////////////////////

    size_t channels = out_mat.channels();
    CV_Assert(1 == channels || 3 == channels);

    int depth = CV_MAT_DEPTH(type);
    CV_Assert(CV_8U == depth || CV_32F == depth);

    CV_Assert(cv::INTER_AREA == interp || cv::INTER_LINEAR == interp);

    ASSERT_TRUE(in_mat1.isContinuous() && out_mat.isContinuous());

    using namespace InferenceEngine;

    size_t  in_height = in_mat1.rows,  in_width = in_mat1.cols;
    size_t out_height = out_mat.rows, out_width = out_mat.cols;
    InferenceEngine::SizeVector  in_sv = { 1, channels,  in_height,  in_width };
    InferenceEngine::SizeVector out_sv = { 1, channels, out_height, out_width };

    // HWC blob: channels are interleaved
    Precision precision = CV_8U == depth ? Precision::U8 : Precision::FP32;
    TensorDesc  in_desc(precision,  in_sv, Layout::NHWC);
    TensorDesc out_desc(precision, out_sv, Layout::NHWC);

    Blob::Ptr in_blob, out_blob;
    in_blob  = make_blob_with_precision(in_desc , in_mat1.data);
    out_blob = make_blob_with_precision(out_desc, out_mat.data);

    PreProcessDataPtr preprocess = CreatePreprocDataHelper();
    preprocess->setRoiBlob(in_blob);

    ResizeAlgorithm algorithm = cv::INTER_AREA == interp ? RESIZE_AREA : RESIZE_BILINEAR;
    PreProcessInfo info;
    info.setResizeAlgorithm(algorithm);

    // test once to warm-up cache
    preprocess->execute(out_blob, info, false);

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ preprocess->execute(out_blob, info, false); },
            100, "Resize IE %s %s %dx%d -> %dx%d",
            interpToString(interp).c_str(), typeToString(type).c_str(),
            sz_in.width, sz_in.height, sz_out.width, sz_out.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::resize(in_mat1, out_mat_ocv, sz_out, 0, 0, interp);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat_ocv, out_mat, cv::NORM_INF), tolerance);
    }
}

TEST_P(ColorConvertTestIE, AccuracyTest)
{
    using namespace InferenceEngine;
    int depth = 0;
    auto in_fmt = ColorFormat::RAW;
    auto out_fmt = ColorFormat::BGR;  // for now, always BGR
    auto in_layout = Layout::ANY;
    auto out_layout = Layout::ANY;
    cv::Size size;
    double tolerance = 0.0;
    std::tie(depth, in_fmt, in_layout, out_layout, size, tolerance) = GetParam();

    int in_type = CV_MAKE_TYPE(depth, numChannels(in_fmt));
    int out_type = CV_MAKE_TYPE(depth, numChannels(out_fmt));

    cv::Mat in_mat1(size, in_type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);

    cv::Mat out_mat(size, out_type);
    cv::Mat out_mat_ocv(size, out_type);

    // Inference Engine code ///////////////////////////////////////////////////

    if (in_fmt != ColorFormat::RAW && in_fmt != ColorFormat::BGR) {
        cv::cvtColor(in_mat1, in_mat1, toCvtColorCode(in_fmt));
    }

    size_t in_channels = in_mat1.channels();
    CV_Assert(3 == in_channels || 4 == in_channels);

    size_t out_channels = out_mat.channels();
    CV_Assert(3 == out_channels || 4 == out_channels);

    CV_Assert(CV_8U == depth || CV_32F == depth);

    ASSERT_TRUE(in_mat1.isContinuous() && out_mat.isContinuous());

    size_t  in_height = in_mat1.rows,  in_width = in_mat1.cols;
    size_t out_height = out_mat.rows, out_width = out_mat.cols;
    InferenceEngine::SizeVector  in_sv = { 1, in_channels,  in_height,  in_width };
    InferenceEngine::SizeVector out_sv = { 1, out_channels, out_height, out_width };

    // HWC blob: channels are interleaved
    Precision precision = CV_8U == depth ? Precision::U8 : Precision::FP32;

    Blob::Ptr in_blob, out_blob;
    switch (precision)
    {
    case Precision::U8:
        in_blob = img2Blob<Precision::U8>(in_mat1, in_layout);
        out_blob = img2Blob<Precision::U8>(out_mat, out_layout);
        break;

    case Precision::FP32:
        in_blob = img2Blob<Precision::FP32>(in_mat1, in_layout);
        out_blob = img2Blob<Precision::FP32>(out_mat, out_layout);
        break;

    default:
        FAIL() << "Unsupported configuration";
    }

    PreProcessDataPtr preprocess = CreatePreprocDataHelper();
    preprocess->setRoiBlob(in_blob);

    PreProcessInfo info;
    info.setColorFormat(in_fmt);

    // test once to warm-up cache
    preprocess->execute(out_blob, info, false);

    switch (precision)
    {
    case Precision::U8:   Blob2Img<Precision::U8>  (out_blob, out_mat, out_layout); break;
    case Precision::FP32: Blob2Img<Precision::FP32>(out_blob, out_mat, out_layout); break;
    default: FAIL() << "Unsupported configuration";
    }

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ preprocess->execute(out_blob, info, false); },
            100, "Color Convert IE %s %s %s %dx%d %s->%s",
            depthToString(depth).c_str(),
            layoutToString(in_layout).c_str(), layoutToString(out_layout).c_str(),
            size.width, size.height,
            colorFormatToString(in_fmt).c_str(), colorFormatToString(out_fmt).c_str());
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        if (in_fmt != out_fmt) {
            cv::cvtColor(in_mat1, out_mat_ocv, toCvtColorCode(in_fmt, out_fmt));
        } else {
            // only reorder is done
            out_mat_ocv = in_mat1;
        }
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat_ocv, out_mat, cv::NORM_INF), tolerance);
    }
}

TEST_P(ColorConvertYUV420TestIE, AccuracyTest)
{
    using namespace InferenceEngine;
    const int depth = CV_8U;
    auto in_fmt = ColorFormat::NV12;
    const auto out_fmt = ColorFormat::BGR;  // for now, always BGR
    const auto in_layout = Layout::NCHW;
    auto out_layout = Layout::ANY;
    cv::Size size;
    double tolerance = 0.0;
    std::tie(in_fmt, out_layout, size, tolerance) = GetParam();

    cv::Mat in_mat_y(size, CV_MAKE_TYPE(depth, 1));
    cv::Mat in_mat_uv(cv::Size(size.width / 2, size.height / 2), CV_MAKE_TYPE(depth, 2));
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat_y, mean, stddev);
    cv::randn(in_mat_uv, mean / 2, stddev / 2);

    int out_type = CV_MAKE_TYPE(depth, numChannels(out_fmt));
    cv::Mat out_mat(size, out_type);
    cv::Mat out_mat_ocv(size, out_type);

    // Inference Engine code ///////////////////////////////////////////////////

    size_t out_channels = out_mat.channels();
    CV_Assert(3 == out_channels || 4 == out_channels);

    ASSERT_TRUE(in_mat_y.isContinuous() && out_mat.isContinuous());

    const Precision precision = Precision::U8;

    auto make_nv12_blob = [&](){
        auto y_blob = img2Blob<Precision::U8>(in_mat_y, Layout::NHWC);
        auto uv_blob = img2Blob<Precision::U8>(in_mat_uv, Layout::NHWC);
        return make_shared_blob<NV12Blob>(y_blob, uv_blob);

    };
    auto make_I420_blob = [&](){
        cv::Mat in_mat_u(cv::Size(size.width / 2, size.height / 2), CV_MAKE_TYPE(depth, 1));
        cv::Mat in_mat_v(cv::Size(size.width / 2, size.height / 2), CV_MAKE_TYPE(depth, 1));

        std::array<cv::Mat, 2> in_uv = {in_mat_u, in_mat_v};
        cv::split(in_mat_uv, in_uv);

        auto y_blob = img2Blob<Precision::U8>(in_mat_y, Layout::NHWC);
        auto u_blob = img2Blob<Precision::U8>(in_mat_u, Layout::NHWC);
        auto v_blob = img2Blob<Precision::U8>(in_mat_v, Layout::NHWC);
        return make_shared_blob<I420Blob>(y_blob, u_blob, v_blob);
    };

    Blob::Ptr in_blob = (in_fmt == ColorFormat::NV12) ?  Blob::Ptr{make_nv12_blob()} :  Blob::Ptr {make_I420_blob()};
    auto out_blob = img2Blob<Precision::U8>(out_mat, out_layout);

    PreProcessDataPtr preprocess = CreatePreprocDataHelper();
    preprocess->setRoiBlob(in_blob);

    PreProcessInfo info;
    info.setColorFormat(in_fmt);

    // test once to warm-up cache
    preprocess->execute(out_blob, info, false);

    Blob2Img<Precision::U8>(out_blob, out_mat, out_layout);

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ preprocess->execute(out_blob, info, false); },
            100, "Color Convert IE %s %s %s %dx%d %s->%s",
            depthToString(depth).c_str(),
            layoutToString(in_layout).c_str(), layoutToString(out_layout).c_str(),
            size.width, size.height,
            colorFormatToString(in_fmt).c_str(), colorFormatToString(out_fmt).c_str());
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        //for both I420 and NV12 use NV12 as I420 is not supported by OCV
        cv::cvtColorTwoPlane(in_mat_y, in_mat_uv, out_mat_ocv, toCvtColorCode(ColorFormat::NV12, out_fmt));
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat_ocv, out_mat, cv::NORM_INF), tolerance);
    }
}

TEST_P(SplitTestIE, AccuracyTest)
{
    const auto params = GetParam();
    int type = std::get<0>(params);
    cv::Size size = std::get<1>(params);
    double tolerance = std::get<2>(params);

    int depth = CV_MAT_DEPTH(type);
    CV_Assert(CV_8U == depth || CV_32F == depth);

    int type1 = CV_MAKE_TYPE(depth, 1);
    int type4 = CV_MAKE_TYPE(depth, 4);

    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::Mat in_mat(size, type);
    cv::randn(in_mat, mean, stddev);

    int channels = in_mat.channels();
    CV_Assert(2 == channels || 3 == channels || 4 == channels);

    size_t elemsize1 = in_mat.elemSize1();
    int    total     = in_mat.total();

    cv::Mat out_mat(size, type4);
    CV_Assert(in_mat.isContinuous() && out_mat.isContinuous());

    cv::Mat out_mat0(size, type1, out_mat.data + 0*total*elemsize1);
    cv::Mat out_mat1(size, type1, out_mat.data + 1*total*elemsize1);
    cv::Mat out_mat2(size, type1, out_mat.data + 2*total*elemsize1);
    cv::Mat out_mat3(size, type1, out_mat.data + 3*total*elemsize1);

    cv::Mat out_mats[] = {out_mat0, out_mat1, out_mat2, out_mat3};

    std::vector<cv::Mat> out_mats_ocv(channels);

    // Inference Engine code ///////////////////////////////////////////////////

    using namespace InferenceEngine;

    size_t width  = size.width;
    size_t height = size.height;
    InferenceEngine::SizeVector sv = { 1, (size_t)channels, height,  width };

    Precision precision = CV_8U == depth ? Precision::U8 : Precision::FP32;
    TensorDesc  in_desc(precision, sv, Layout::NHWC); // interleaved
    TensorDesc out_desc(precision, sv, Layout::NCHW); // color planes

    Blob::Ptr in_blob, out_blob;
    in_blob  = make_blob_with_precision( in_desc,  in_mat.data);
    out_blob = make_blob_with_precision(out_desc, out_mat.data);

    // test once
    blob_copy(in_blob, out_blob);

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&]() { blob_copy(in_blob, out_blob); },
        400, "Split IE %s %dx%d", typeToString(type).c_str(), size.width, size.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////

    cv::split(in_mat, out_mats_ocv);

    // Comparison //////////////////////////////////////////////////////////////

    for (int i = 0; i < channels; i++)
    {
        EXPECT_LE(cv::norm(out_mats_ocv[i], out_mats[i], cv::NORM_INF), tolerance);
    }
}

TEST_P(MergeTestIE, AccuracyTest)
{
    const auto params = GetParam();
    int type = std::get<0>(params);
    cv::Size size = std::get<1>(params);
    double tolerance = std::get<2>(params);

    int depth = CV_MAT_DEPTH(type);
    CV_Assert(CV_8U == depth || CV_32F == depth);

    int type1 = CV_MAKE_TYPE(depth, 1);
    int type4 = CV_MAKE_TYPE(depth, 4);

    cv::Mat out_mat(size, type), out_mat_ocv;

    cv::Mat in_mat(size, type4);

    int channels = out_mat.channels();
    CV_Assert(2 == channels || 3 == channels || 4 == channels);

    size_t elemsize1 = out_mat.elemSize1();
    int    total     = out_mat.total();

    cv::Mat in_mat0(size, type1, in_mat.data + 0*total*elemsize1);
    cv::Mat in_mat1(size, type1, in_mat.data + 1*total*elemsize1);
    cv::Mat in_mat2(size, type1, in_mat.data + 2*total*elemsize1);
    cv::Mat in_mat3(size, type1, in_mat.data + 3*total*elemsize1);

    cv::Mat in_mats[] = { in_mat0, in_mat1, in_mat2, in_mat3 };

    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    for (int i = 0; i < 4 ; i++)
    {
        cv::randn(in_mats[i], mean, stddev);
    }

    CV_Assert(in_mat.isContinuous() && out_mat.isContinuous());

    // Inference Engine code ///////////////////////////////////////////////////

    using namespace InferenceEngine;

    size_t width  = size.width;
    size_t height = size.height;
    InferenceEngine::SizeVector sv = { 1, (size_t)channels, height,  width };

    Precision precision = CV_8U == depth ? Precision::U8 : Precision::FP32;
    TensorDesc  in_desc(precision, sv, Layout::NCHW); // color planes
    TensorDesc out_desc(precision, sv, Layout::NHWC); // interleaved

    Blob::Ptr in_blob, out_blob;
    in_blob  = make_blob_with_precision( in_desc,  in_mat.data);
    out_blob = make_blob_with_precision(out_desc, out_mat.data);

    // test once
    blob_copy(in_blob, out_blob);

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&]() { blob_copy(in_blob, out_blob); },
        400, "Merge IE %s %dx%d", typeToString(type).c_str(), size.width, size.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////

    cv::merge(in_mats, channels, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////

    EXPECT_LE(cv::norm(out_mat_ocv, out_mat, cv::NORM_INF), tolerance);
}

TEST_P(PreprocTest, Performance)
{
    using namespace InferenceEngine;
    Precision prec;
    ResizeAlgorithm interp;
    Layout in_layout, out_layout;
    std::pair<int, int> ocv_channels{-1, -1};
    std::pair<cv::Size, cv::Size> sizes;
    ColorFormat in_fmt = ColorFormat::RAW;
    ColorFormat out_fmt = ColorFormat::BGR;
    std::tie(prec, interp, in_fmt, in_layout, out_layout, ocv_channels, sizes) = GetParam();
    cv::Size in_size, out_size;
    std::tie(in_size, out_size) = sizes;
    int in_ocv_chan = -1, out_ocv_chan = -1;
    std::tie(in_ocv_chan, out_ocv_chan) = ocv_channels;
#if defined(__arm__) || defined(__aarch64__)
    double tolerance = Precision::U8 ? 4 : 0.015;
#else
    double tolerance = Precision::U8 ? 1 : 0.015;
#endif

    const int ocv_depth = prec == Precision::U8 ? CV_8U :
        prec == Precision::FP32 ? CV_32F : -1;
    const int in_ocv_type = CV_MAKETYPE(ocv_depth, in_ocv_chan);
    const int out_ocv_type = CV_MAKETYPE(ocv_depth, out_ocv_chan);
    initMatrixRandU(in_ocv_type, in_size, in_ocv_type, false);

    cv::Mat out_mat(out_size, out_ocv_type);

    // convert input mat to correct color format if required. note that NV12 being a planar case is
    // handled separately
    if (in_fmt != ColorFormat::RAW && in_fmt != ColorFormat::BGR && in_fmt != ColorFormat::NV12) {
        cv::cvtColor(in_mat1, in_mat1, toCvtColorCode(in_fmt));
    }
    // create additional cv::Mat in NV12 case
    if (in_fmt == ColorFormat::NV12) {
        in_mat2 = cv::Mat(cv::Size(in_mat1.cols / 2, in_mat1.rows / 2), CV_8UC2);
        cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));
    }

    Blob::Ptr in_blob, out_blob;
    switch (prec)
    {
    case Precision::U8:
        if (in_fmt == ColorFormat::NV12) {
            auto y_blob = img2Blob<Precision::U8>(in_mat1, Layout::NHWC);
            auto uv_blob = img2Blob<Precision::U8>(in_mat2, Layout::NHWC);
            in_blob = make_shared_blob<NV12Blob>(y_blob, uv_blob);
        } else {
            in_blob = img2Blob<Precision::U8>(in_mat1, in_layout);
        }
        out_blob = img2Blob<Precision::U8>(out_mat, out_layout);
        break;

    case Precision::FP32:
        in_blob = img2Blob<Precision::FP32>(in_mat1, in_layout);
        out_blob = img2Blob<Precision::FP32>(out_mat, out_layout);
        break;

    default:
        FAIL() << "Unsupported configuration";
    }

    PreProcessDataPtr preprocess = CreatePreprocDataHelper();
    preprocess->setRoiBlob(in_blob);

    PreProcessInfo info;
    info.setResizeAlgorithm(interp);
    info.setColorFormat(in_fmt);

    // test once to warm-up cache
    preprocess->execute(out_blob, info, false);

    switch (prec)
    {
    case Precision::U8:   Blob2Img<Precision::U8>  (out_blob, out_mat, out_layout); break;
    case Precision::FP32: Blob2Img<Precision::FP32>(out_blob, out_mat, out_layout); break;
    default: FAIL() << "Unsupported configuration";
    }

    cv::Mat ocv_out_mat(in_mat1);

    if (in_fmt != ColorFormat::RAW && in_fmt != out_fmt && in_fmt != ColorFormat::NV12) {
        cv::cvtColor(ocv_out_mat, ocv_out_mat, toCvtColorCode(in_fmt, out_fmt));
    } else if (in_fmt == ColorFormat::NV12) {
        cv::cvtColorTwoPlane(ocv_out_mat, in_mat2, ocv_out_mat, toCvtColorCode(in_fmt, out_fmt));
    }

    auto cv_interp = interp == RESIZE_AREA ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(ocv_out_mat, ocv_out_mat, out_size, 0, 0, cv_interp);

    EXPECT_LE(cv::norm(ocv_out_mat, out_mat, cv::NORM_INF), tolerance);

#if PERF_TEST
    // iterate testing, and print performance
    const auto type_str = depthToString(ocv_depth);
    const auto interp_str = interp == RESIZE_AREA ? "AREA"
        : interp == RESIZE_BILINEAR ? "BILINEAR" : "?";
    const auto in_layout_str = layoutToString(in_layout);
    const auto out_layout_str = layoutToString(out_layout);

    test_ms([&]() { preprocess->execute(out_blob, info, false); },
            300,
            "Preproc %s %s %d %s %dx%d %d %s %dx%d %s->%s",
            type_str.c_str(),
            interp_str,
            in_ocv_chan,
            in_layout_str.c_str(), in_size.width, in_size.height,
            out_ocv_chan,
            out_layout_str.c_str(), out_size.width, out_size.height,
            colorFormatToString(in_fmt).c_str(), colorFormatToString(out_fmt).c_str());
#endif // PERF_TEST

}
