// Copyright (C) 2018-2023 Intel Corporation
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

#include <stdexcept>

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
    case CV_16U : return "CV_16U";
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
    case CV_8SC1  : return "CV_8SC1";
    case CV_8SC2  : return "CV_8SC2";
    case CV_8SC3  : return "CV_8SC3";
    case CV_8SC4  : return "CV_8SC4";
    case CV_16SC1 : return "CV_16SC1";
    case CV_16SC2 : return "CV_16SC2";
    case CV_16SC3 : return "CV_16SC3";
    case CV_16SC4 : return "CV_16SC4";
    case CV_16UC1 : return "CV_16UC1";
    case CV_16UC2 : return "CV_16UC2";
    case CV_16UC3 : return "CV_16UC3";
    case CV_16UC4 : return "CV_16UC4";
    case CV_32SC1 : return "CV_32SC1";
    case CV_32SC2 : return "CV_32SC2";
    case CV_32SC3 : return "CV_32SC3";
    case CV_32SC4 : return "CV_32SC4";
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
        default: IE_THROW() << "Unrecognized color format";
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
    for (size_t i = 0; i < mats.size(); i++) {
        test_mats[i] = to_test(mats[i]);
    }
    return test_mats;
}

test::Rect to_test(cv::Rect& rect) { return {rect.x, rect.y, rect.width, rect.height}; }
test::Scalar to_test(cv::Scalar const& sc) { return {sc[0], sc[1], sc[2], sc[3]}; }

cv::ColorConversionCodes toCvtColorCode(InferenceEngine::ColorFormat in,
                                     InferenceEngine::ColorFormat out) {
    using namespace InferenceEngine;
    IE_SUPPRESS_DEPRECATED_START
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
    };
    IE_SUPPRESS_DEPRECATED_END
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
        default: IE_THROW() << "Unrecognized color format";
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

    CV_Assert(cv::DataType<data_t>::depth == img.depth() || (PRC == Precision::FP16 && img.depth() == CV_16F));

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
            IE_THROW() << "Inconsistent input layout for image processing: " << layout;
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

    //IE and OpenCV use different data types for FP16 representation, so need to check for it explicitly
    CV_Assert(cv::DataType<data_t>::depth == img.depth() || ((img.depth() == CV_16F) && (PRC == Precision::FP16)));

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
            IE_THROW() << "Inconsistent input layout for image processing: " << layout;
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

    auto make_src_type = [planes](int d){
            return CV_MAKE_TYPE(d, planes);
    };
    int srcType = make_src_type(depth);
    int dstType = CV_MAKE_TYPE(depth, 1);

    cv::Mat in_mat(sz, srcType);
    bool const is_fp16 = (depth == CV_16F);
    cv::Mat rnd_mat =  is_fp16 ? cv::Mat(sz, make_src_type(CV_32F)) : in_mat;
    cv::randn(rnd_mat, cv::Scalar::all(127), cv::Scalar::all(40.f));

    if (is_fp16) {
        rnd_mat.convertTo(in_mat, depth);
    }

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

    auto make_src_type = [](int d){
            return CV_MAKE_TYPE(d, 1);
    };
    int srcType = make_src_type(depth);
    int dstType = CV_MAKE_TYPE(depth, planes);

    std::vector<cv::Mat> in_mats(planes, cv::Mat(sz, srcType));
    for (int p = 0; p < planes; p++) {
        bool const is_fp16 = (depth == CV_16F);
        cv::Mat rnd_mat =  is_fp16 ? cv::Mat(sz, make_src_type(CV_32F)) : in_mats[p];
        cv::randn(rnd_mat, cv::Scalar::all(127), cv::Scalar::all(40.f));

        if (is_fp16) {
            rnd_mat.convertTo(in_mats[p], depth);
        }
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

TEST_P(ConvertDepthTestGAPI, AccuracyTest)
{
    const auto params = GetParam();
    int in_depth      = std::get<0>(params);
    int out_depth     = std::get<1>(params);
    cv::Size sz       = std::get<2>(params);
    double tolerance  = std::get<3>(params);

    const int out_type = CV_MAKETYPE(out_depth,1);

    initMatrixRandU(CV_MAKETYPE(in_depth,1), sz, out_type);

    // G-API code //////////////////////////////////////////////////////////////
    ConvertDepthComputation cc(to_test(in_mat1), to_test(out_mat_gapi), out_mat_gapi.depth());
    cc.warmUp();

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ cc.apply(); },
        400, "ConvDepth GAPI %s to %s %dx%d", depthToString(in_mat1.depth()).c_str(), depthToString(out_mat_gapi.depth()).c_str(), sz.width, sz.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        in_mat1.convertTo(out_mat_ocv, out_type);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat_ocv, out_mat_gapi, cv::NORM_INF), tolerance);
    }
}

TEST_P(DivCTestGAPI, AccuracyTest)
{
    const auto params       = GetParam();
    const int in_depth      = std::get<0>(params);
    const int in_channels   = std::get<1>(params);
    const cv::Size sz       = std::get<2>(params);
    const cv::Scalar C      = std::get<3>(params);
    double tolerance        = std::get<4>(params);

    const int in_type = CV_MAKETYPE(in_depth,in_channels);

    initMatrixRandU(in_type, sz, in_type);

    // G-API code
    DivCComputation cc(to_test(in_mat1), to_test(out_mat_gapi), to_test(C));
    cc.warmUp();

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_mat_ocv = in_mat1 / C;
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat_ocv, out_mat_gapi, cv::NORM_INF), tolerance);
    }
}

TEST_P(SubCTestGAPI, AccuracyTest)
{
    const auto params       = GetParam();
    const int in_depth      = std::get<0>(params);
    const int in_channels   = std::get<1>(params);
    const cv::Size sz       = std::get<2>(params);
    const cv::Scalar C      = std::get<3>(params);
    const double tolerance  = std::get<4>(params);

    const int in_type = CV_MAKETYPE(in_depth,in_channels);

    initMatrixRandU(in_type, sz, in_type);

    // G-API code
    SubCComputation cc(to_test(in_mat1), to_test(out_mat_gapi), to_test(C));
    cc.warmUp();

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_mat_ocv = in_mat1 - C;
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

IE_SUPPRESS_DEPRECATED_START

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

    if (depth != CV_16F)
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

    CV_Assert(CV_8U == depth || CV_32F == depth || depth == CV_16S || depth == CV_16F);

    ASSERT_TRUE(in_mat1.isContinuous() && out_mat.isContinuous());

    size_t  in_height = in_mat1.rows,  in_width = in_mat1.cols;
    size_t out_height = out_mat.rows, out_width = out_mat.cols;
    InferenceEngine::SizeVector  in_sv = { 1, in_channels,  in_height,  in_width };
    InferenceEngine::SizeVector out_sv = { 1, out_channels, out_height, out_width };

    auto depth_to_precision = [](int depth) -> Precision::ePrecision {
        switch (depth)
        {
            case CV_8U:  return Precision::U8;
            case CV_16S: return Precision::I16;
            case CV_16F: return Precision::FP16;
            case CV_32F: return Precision::FP32;
            default:
                throw std::logic_error("Unsupported configuration");
        }
        return Precision::UNSPECIFIED;
    };

    // HWC blob: channels are interleaved
    Precision precision = depth_to_precision(depth);

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

    case Precision::I16:
        in_blob = img2Blob<Precision::I16>(in_mat1, in_layout);
        out_blob = img2Blob<Precision::I16>(out_mat, out_layout);
        break;

    case Precision::FP16:
        in_blob =  img2Blob<Precision::FP16>(in_mat1, in_layout);
        out_blob = img2Blob<Precision::FP16>(out_mat, out_layout);

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
    case Precision::I16:  Blob2Img<Precision::I16> (out_blob, out_mat, out_layout); break;
    case Precision::FP16: Blob2Img<Precision::FP16> (out_blob, out_mat, out_layout); break;
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

TEST_P(MeanValueGAPI, AccuracyTest)
{
    const auto params = GetParam();
    cv::Size sz       = std::get<0>(params);
    double tolerance  = std::get<1>(params);

    initMatrixRandU(CV_32FC1, sz, CV_32FC1);

    const cv::Scalar mean = { 0.485, 0.456, 0.406 };
    const cv::Scalar std  = { 0.229, 0.224, 0.225 };

    // G-API code
    MeanValueSubtractComputation cc(to_test(in_mat1), to_test(out_mat_gapi), to_test(mean), to_test(std));
    cc.warmUp();

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_mat_ocv = (in_mat1 - mean) / std;
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(cv::norm(out_mat_ocv, out_mat_gapi, cv::NORM_INF), tolerance);
    }

}

