// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENCV_GAPI_CORE_TESTS_INL_HPP
#define OPENCV_GAPI_CORE_TESTS_INL_HPP

#include "gapi_core_tests.hpp"

#include "blob_factory.hpp"
#include "blob_transform.hpp"
#include "ie_preprocess_data.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi.hpp>

#include <cstdarg>
#include <cstdio>
#include <ctime>

#include <chrono>

#include <fluid_test_computations.hpp>

// Can be set externally (via CMake) if built with -DGAPI_TEST_PERF=ON
#ifndef PERF_TEST
#define PERF_TEST 0 // 1=test performance, 0=don't
#endif

namespace opencv_test
{

#if PERF_TEST
// performance test: iterate function, measure and print milliseconds per call
template<typename F> static void test_ms(F func, int iter, const char format[], ...)
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

static cv::String interpToString(int interp)
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

static cv::String depthToString(int depth)
{
    switch(depth)
    {
    case CV_8U  : return "CV_8U";
    case CV_32F : return "CV_32F";
    }
    CV_Assert(!"ERROR: unsupported depth!");
    return nullptr;
}

static cv::String typeToString(int type)
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
#endif  // PERF_TEST

namespace {

test::Mat to_test(cv::Mat& mat) { return {mat.rows, mat.cols, mat.type(), mat.data}; }
std::vector<test::Mat> to_test(std::vector<cv::Mat>& mats)
{
    std::vector<test::Mat> test_mats(mats.size());
    for (int i = 0; i < mats.size(); i++) {
        test_mats[i] = to_test(mats[i]);
    }
    return test_mats;
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
        cv::Mat absDiff;
        cv::absdiff(out_mat, out_mat_ocv, absDiff);
        EXPECT_EQ(0, cv::countNonZero(absDiff > tolerance));
    }
}

TEST_P(SplitTestGAPI, AccuracyTest)
{
    const auto params = GetParam();
    int planes  = std::get<0>(params);
    int depth   = std::get<1>(params);
    cv::Size sz = std::get<2>(params);

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
            EXPECT_EQ(0, cv::countNonZero(out_mats_ocv[p]  != out_mats_gapi[p]));
        }
    }
}

TEST_P(MergeTestGAPI, AccuracyTest)
{
    const auto params = GetParam();
    int planes  = std::get<0>(params);
    int depth   = std::get<1>(params);
    cv::Size sz = std::get<2>(params);

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
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
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

    PreProcessData preprocess;
    preprocess.setRoiBlob(in_blob);

    ResizeAlgorithm algorithm = cv::INTER_AREA == interp ? RESIZE_AREA : RESIZE_BILINEAR;

    // test once to warm-up cache
    preprocess.execute(out_blob, algorithm, false);

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ preprocess.execute(out_blob, algorithm, false); },
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
        cv::Mat absDiff;
        cv::absdiff(out_mat, out_mat_ocv, absDiff);
        EXPECT_EQ(0, cv::countNonZero(absDiff > tolerance));
    }
}

TEST_P(SplitTestIE, AccuracyTest)
{
    int type = std::get<0>(GetParam());
    cv::Size size = std::get<1>(GetParam());

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
        EXPECT_EQ(0, cv::countNonZero(out_mats[i] != out_mats_ocv[i]));
    }
}

TEST_P(MergeTestIE, AccuracyTest)
{
    int type = std::get<0>(GetParam());
    cv::Size size = std::get<1>(GetParam());

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

    EXPECT_EQ(0, cv::countNonZero(out_mat != out_mat_ocv));
}

namespace
{
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
}  // namespace

TEST_P(PreprocTest, Performance)
{
    using namespace InferenceEngine;
    Precision prec;
    ResizeAlgorithm interp;
    Layout in_layout, out_layout;
    int ocv_chan = -1;
    std::pair<cv::Size, cv::Size> sizes;
    std::tie(prec, interp, in_layout, out_layout, ocv_chan, sizes) = GetParam();
    cv::Size in_size, out_size;
    std::tie(in_size, out_size) = sizes;

    const int ocv_depth = prec == Precision::U8 ? CV_8U :
        prec == Precision::FP32 ? CV_32F : -1;
    const int ocv_type = CV_MAKETYPE(ocv_depth, ocv_chan);
    initMatrixRandU(ocv_type, in_size, ocv_type, false);

    cv::Mat out_mat(out_size, ocv_type);

    Blob::Ptr in_blob, out_blob;
    switch (prec)
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

    PreProcessData preprocess;
    preprocess.setRoiBlob(in_blob);

    // test once to warm-up cache
    preprocess.execute(out_blob, interp, false);

    switch (prec)
    {
    case Precision::U8:   Blob2Img<Precision::U8>  (out_blob, out_mat, out_layout); break;
    case Precision::FP32: Blob2Img<Precision::FP32>(out_blob, out_mat, out_layout); break;
    default: FAIL() << "Unsupported configuration";
    }

    cv::Mat ocv_out_mat(out_size, ocv_type);
    auto cv_interp = interp == RESIZE_AREA ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(in_mat1, ocv_out_mat, out_size, 0, 0, cv_interp);

    cv::Mat absDiff;
    cv::absdiff(ocv_out_mat, out_mat, absDiff);
    EXPECT_EQ(cv::countNonZero(absDiff > 1), 0);

#if PERF_TEST
    // iterate testing, and print performance
    const auto type_str = depthToString(ocv_depth);
    const auto interp_str = interp == RESIZE_AREA ? "AREA"
        : interp == RESIZE_BILINEAR ? "BILINEAR" : "?";
    const auto layout_to_str = [](const Layout &l) {
        switch (l) {
        case Layout::NCHW: return "NCHW";
        case Layout::NHWC: return "NHWC";
        default: return "?";
        }
    };
    const auto in_layout_str = layout_to_str(in_layout);
    const auto out_layout_str = layout_to_str(out_layout);

    test_ms([&]() { preprocess.execute(out_blob, interp, false); },
            300,
            "Preproc %s %d %s %s %dx%d %s %dx%d",
            type_str.c_str(),
            ocv_chan,
            interp_str,
            in_layout_str, in_size.width, in_size.height,
            out_layout_str, out_size.width, out_size.height);
#endif // PERF_TEST

}

} // opencv_test

#endif //OPENCV_GAPI_CORE_TESTS_INL_HPP
