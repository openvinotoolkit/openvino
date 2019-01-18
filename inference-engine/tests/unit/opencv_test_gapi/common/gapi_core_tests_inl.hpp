// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENCV_GAPI_CORE_TESTS_INL_HPP
#define OPENCV_GAPI_CORE_TESTS_INL_HPP

#include "gapi_core_tests.hpp"

#include "blob_factory.hpp"
#include "blob_transform.hpp"
#include "ie_preprocess.hpp"
#include "ie_preprocess_data.hpp"
#include "ie_preprocess_gapi_kernels.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi.hpp>

#include <cstdarg>
#include <cstdio>
#include <ctime>

#include <chrono>

#define CV_MAT_CHANNELS(flags) (((flags) >> CV_CN_SHIFT) + 1)

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

TEST_P(ResizeTestGAPI, AccuracyTest)
{
    int type = 0, interp = 0;
    cv::Size sz_in, sz_out;
    double tolerance = 0.0;
    cv::GCompileArgs compile_args;
    std::pair<cv::Size, cv::Size> sizes;
    std::tie(type, interp, sizes, tolerance, compile_args) = GetParam();
    std::tie(sz_in, sz_out) = sizes;

    cv::Mat in_mat1 (sz_in, type );
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    cv::Mat out_mat_ocv(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in, out;
    switch (CV_MAT_CHANNELS(type))
    {
    case 1:
        out = InferenceEngine::gapi::ScalePlane::on(in, type, sz_in, sz_out, interp);
        break;
    case 3:
        {
        int depth = CV_MAT_DEPTH(type);
        int type1 = CV_MAKE_TYPE(depth, 1);
        cv::GMat in0, in1, in2, out0, out1, out2;
        std::tie(in0, in1, in2) = InferenceEngine::gapi::Split3::on(in);
        out0 = InferenceEngine::gapi::ScalePlane::on(in0, type1, sz_in, sz_out, interp);
        out1 = InferenceEngine::gapi::ScalePlane::on(in1, type1, sz_in, sz_out, interp);
        out2 = InferenceEngine::gapi::ScalePlane::on(in2, type1, sz_in, sz_out, interp);
        out = InferenceEngine::gapi::Merge3::on(out0, out1, out2);
        }
        break;
    default: CV_Assert(!"ERROR: unsupported number of channels!");
    }

    cv::GComputation c(in, out);

    // compile graph, and test once

    auto own_in_mat1 = cv::to_own(in_mat1);
    auto own_out_mat = cv::to_own(out_mat);

    std::vector<cv::gapi::own::Mat> v_in  = { own_in_mat1 };
    std::vector<cv::gapi::own::Mat> v_out = { own_out_mat };

    c.apply(v_in, v_out, std::move(compile_args));

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ c.apply(v_in, v_out); },
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

TEST_P(Split2TestGAPI, AccuracyTest)
{
    int depth = std::get<0>(GetParam());
    cv::Size sz_in = std::get<1>(GetParam());
    auto compile_args = std::get<2>(GetParam());

    int type1 = CV_MAKE_TYPE(depth, 1);
    int type2 = CV_MAKE_TYPE(depth, 2);
    initMatrixRandU(type2, sz_in, type1);

    cv::Mat out_mat2 = cv::Mat(sz_in, type1);
    cv::Mat out_mat_ocv2 = cv::Mat(sz_in, type1);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2;
    std::tie(out1, out2) = InferenceEngine::gapi::Split2::on(in1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2));

    // compile graph, and test once

    auto own_in_mat1      = cv::to_own(in_mat1);
    auto own_out_mat_gapi = cv::to_own(out_mat_gapi);
    auto own_out_mat2     = cv::to_own(out_mat2);

    std::vector<cv::gapi::own::Mat> v_in  = { own_in_mat1 };
    std::vector<cv::gapi::own::Mat> v_out = { own_out_mat_gapi, own_out_mat2 };

    c.apply(v_in, v_out, std::move(compile_args));

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ c.apply(v_in, v_out); },
        400, "Split GAPI %s %dx%d", typeToString(type2).c_str(), sz_in.width, sz_in.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> out_mats_ocv = {out_mat_ocv, out_mat_ocv2};
        cv::split(in_mat1, out_mats_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv  != out_mat_gapi));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv2 != out_mat2));
    }
}

TEST_P(Split3TestGAPI, AccuracyTest)
{
    int depth = std::get<0>(GetParam());
    cv::Size sz_in = std::get<1>(GetParam());
    auto compile_args = std::get<2>(GetParam());

    int type1 = CV_MAKE_TYPE(depth, 1);
    int type3 = CV_MAKE_TYPE(depth, 3);
    initMatrixRandU(type3, sz_in, type1);

    cv::Mat out_mat2 = cv::Mat(sz_in, type1);
    cv::Mat out_mat3 = cv::Mat(sz_in, type1);
    cv::Mat out_mat_ocv2 = cv::Mat(sz_in, type1);
    cv::Mat out_mat_ocv3 = cv::Mat(sz_in, type1);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2, out3;
    std::tie(out1, out2, out3) = InferenceEngine::gapi::Split3::on(in1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2, out3));

    // compile graph, and test once

    auto own_in_mat1      = cv::to_own(in_mat1);
    auto own_out_mat_gapi = cv::to_own(out_mat_gapi);
    auto own_out_mat2     = cv::to_own(out_mat2);
    auto own_out_mat3     = cv::to_own(out_mat3);

    std::vector<cv::gapi::own::Mat> v_in  = { own_in_mat1 };
    std::vector<cv::gapi::own::Mat> v_out = { own_out_mat_gapi, own_out_mat2, own_out_mat3 };

    c.apply(v_in, v_out, std::move(compile_args));

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ c.apply(v_in, v_out); },
        400, "Split GAPI %s %dx%d", typeToString(type3).c_str(), sz_in.width, sz_in.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> out_mats_ocv = {out_mat_ocv, out_mat_ocv2, out_mat_ocv3};
        cv::split(in_mat1, out_mats_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv  != out_mat_gapi));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv2 != out_mat2));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv3 != out_mat3));
    }
}

TEST_P(Split4TestGAPI, AccuracyTest)
{
    int depth = std::get<0>(GetParam());
    cv::Size sz_in = std::get<1>(GetParam());
    auto compile_args = std::get<2>(GetParam());

    int type1 = CV_MAKE_TYPE(depth, 1);
    int type4 = CV_MAKE_TYPE(depth, 4);
    initMatrixRandU(type4, sz_in, type1);

    cv::Mat out_mat2 = cv::Mat(sz_in, type1);
    cv::Mat out_mat3 = cv::Mat(sz_in, type1);
    cv::Mat out_mat4 = cv::Mat(sz_in, type1);
    cv::Mat out_mat_ocv2 = cv::Mat(sz_in, type1);
    cv::Mat out_mat_ocv3 = cv::Mat(sz_in, type1);
    cv::Mat out_mat_ocv4 = cv::Mat(sz_in, type1);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2, out3, out4;
    std::tie(out1, out2, out3, out4) = InferenceEngine::gapi::Split4::on(in1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2, out3, out4));

    // compile graph, and test once

    auto own_in_mat1      = cv::to_own(in_mat1);
    auto own_out_mat_gapi = cv::to_own(out_mat_gapi);
    auto own_out_mat2     = cv::to_own(out_mat2);
    auto own_out_mat3     = cv::to_own(out_mat3);
    auto own_out_mat4     = cv::to_own(out_mat4);

    std::vector<cv::gapi::own::Mat> v_in  = { own_in_mat1 };
    std::vector<cv::gapi::own::Mat> v_out = { own_out_mat_gapi, own_out_mat2,
                                                  own_out_mat3, own_out_mat4 };

    c.apply(v_in, v_out, std::move(compile_args));

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ c.apply(v_in, v_out); },
        400, "Split GAPI %s %dx%d", typeToString(type4).c_str(), sz_in.width, sz_in.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> out_mats_ocv = {out_mat_ocv, out_mat_ocv2, out_mat_ocv3, out_mat_ocv4};
        cv::split(in_mat1, out_mats_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv  != out_mat_gapi));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv2 != out_mat2));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv3 != out_mat3));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv4 != out_mat4));
    }
}

TEST_P(Merge2TestGAPI, AccuracyTest)
{
    int depth = std::get<0>(GetParam());
    cv::Size sz_in = std::get<1>(GetParam());
    auto compile_args = std::get<2>(GetParam());

    int type1 = CV_MAKE_TYPE(depth, 1);
    int type2 = CV_MAKE_TYPE(depth, 2);
    initMatsRandU(type1, sz_in, type2);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = InferenceEngine::gapi::Merge2::on(in1, in2);
    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(out));

    // compile graph, and test once

    auto own_in_mat1      = cv::to_own(in_mat1);
    auto own_in_mat2      = cv::to_own(in_mat2);
    auto own_out_mat_gapi = cv::to_own(out_mat_gapi);

    std::vector<cv::gapi::own::Mat> v_in  = { own_in_mat1, own_in_mat2 };
    std::vector<cv::gapi::own::Mat> v_out = { own_out_mat_gapi };

    c.apply(v_in, v_out, std::move(compile_args));

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ c.apply(v_in, v_out); },
        400, "Merge GAPI %s %dx%d", typeToString(type2).c_str(), sz_in.width, sz_in.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> in_mats_ocv = {in_mat1, in_mat2};
        cv::merge(in_mats_ocv, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
    }
}

TEST_P(Merge3TestGAPI, AccuracyTest)
{
    int depth = std::get<0>(GetParam());
    cv::Size sz_in = std::get<1>(GetParam());
    auto compile_args = std::get<2>(GetParam());

    int type1 = CV_MAKE_TYPE(depth, 1);
    int type3 = CV_MAKE_TYPE(depth, 3);
    initMatsRandU(type1, sz_in, type3);

    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::Mat in_mat3(sz_in,  type1);
    cv::randn(in_mat3, mean, stddev);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, in3;
    auto out = InferenceEngine::gapi::Merge3::on(in1, in2, in3);
    cv::GComputation c(cv::GIn(in1, in2, in3), cv::GOut(out));

    // compile graph, and test once

    auto own_in_mat1      = cv::to_own(in_mat1);
    auto own_in_mat2      = cv::to_own(in_mat2);
    auto own_in_mat3      = cv::to_own(in_mat3);
    auto own_out_mat_gapi = cv::to_own(out_mat_gapi);

    std::vector<cv::gapi::own::Mat> v_in  = { own_in_mat1, own_in_mat2, own_in_mat3 };
    std::vector<cv::gapi::own::Mat> v_out = { own_out_mat_gapi };

    c.apply(v_in, v_out, std::move(compile_args));

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ c.apply(v_in, v_out); },
        400, "Merge GAPI %s %dx%d", typeToString(type3).c_str(), sz_in.width, sz_in.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> in_mats_ocv = {in_mat1, in_mat2, in_mat3};
        cv::merge(in_mats_ocv, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
    }
}

TEST_P(Merge4TestGAPI, AccuracyTest)
{
    int depth = std::get<0>(GetParam());
    cv::Size sz_in = std::get<1>(GetParam());
    auto compile_args = std::get<2>(GetParam());

    int type1 = CV_MAKE_TYPE(depth, 1);
    int type4 = CV_MAKE_TYPE(depth, 4);
    initMatsRandU(type1, sz_in, type4);

    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::Mat in_mat3(sz_in,  type1);
    cv::Mat in_mat4(sz_in,  type1);
    cv::randn(in_mat3, mean, stddev);
    cv::randn(in_mat4, mean, stddev);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, in3, in4;
    auto out = InferenceEngine::gapi::Merge4::on(in1, in2, in3, in4);
    cv::GComputation c(cv::GIn(in1, in2, in3, in4), cv::GOut(out));

    // compile graph, and test once

    auto own_in_mat1      = cv::to_own(in_mat1);
    auto own_in_mat2      = cv::to_own(in_mat2);
    auto own_in_mat3      = cv::to_own(in_mat3);
    auto own_in_mat4      = cv::to_own(in_mat4);
    auto own_out_mat_gapi = cv::to_own(out_mat_gapi);

    std::vector<cv::gapi::own::Mat> v_in  = { own_in_mat1, own_in_mat2, own_in_mat3, own_in_mat4 };
    std::vector<cv::gapi::own::Mat> v_out = { own_out_mat_gapi };

    c.apply(v_in, v_out, std::move(compile_args));

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ c.apply(v_in, v_out); },
        400, "Merge GAPI %s %dx%d", typeToString(type4).c_str(), sz_in.width, sz_in.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> in_mats_ocv = {in_mat1, in_mat2, in_mat3, in_mat4};
        cv::merge(in_mats_ocv, out_mat_ocv);
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
    preprocess.execute(out_blob, algorithm);

#if PERF_TEST
    // iterate testing, and print performance
    test_ms([&](){ preprocess.execute(out_blob, algorithm); },
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
    preprocess.execute(out_blob, interp);

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

    test_ms([&]() { preprocess.execute(out_blob, interp); },
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
