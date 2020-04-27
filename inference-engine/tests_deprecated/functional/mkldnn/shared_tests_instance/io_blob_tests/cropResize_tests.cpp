// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cropResize_tests.hpp"

#ifdef USE_OPENCV

#define COMBINE_WITH_DEFAULT(_dims, _in_layouts, _color_formats) \
    Combine(Values(Precision::FP32), \
            Values(_dims), \
            Values(std::make_pair(Precision::FP32, 1e-2), std::make_pair(Precision::U8, 1)), \
            Values(_in_layouts), \
            Values(ResizeAlgorithm::RESIZE_BILINEAR, ResizeAlgorithm::RESIZE_AREA), \
            Values(_color_formats), \
            Values(ROI({0, 40, 50, 220, 220})), \
            Values(false, true))

// test resize-only for all dims (as before)
// test resize + color conversion for smaller number of dims (simple upscale/downscale scenarios only)
namespace smoke {
static auto params_resize_only = COMBINE_WITH_DEFAULT(
    TESTED_DIMS(1),
    MULTI_VALUE(NCHW, NHWC),
    COLOR_FORMATS_RAW);

static auto params_csc_3ch_and_resize = COMBINE_WITH_DEFAULT(
    TESTED_DIMS_SMALL(1),
    MULTI_VALUE(NCHW, NHWC),
    COLOR_FORMATS_3CH);

static auto params_csc_4ch_and_resize = COMBINE_WITH_DEFAULT(
    TESTED_DIMS_SMALL(1),
    NHWC,
    COLOR_FORMATS_4CH);

// batch preprocessing parameters:
static auto batch_params_resize_only = COMBINE_WITH_DEFAULT(
    TESTED_DIMS(2),
    MULTI_VALUE(NCHW, NHWC),
    COLOR_FORMATS_RAW);

static auto batch_params_csc_3ch_and_resize = COMBINE_WITH_DEFAULT(
    TESTED_DIMS_SMALL(2),
    MULTI_VALUE(NCHW, NHWC),
    COLOR_FORMATS_3CH);

static auto batch_params_csc_4ch_and_resize = COMBINE_WITH_DEFAULT(
    TESTED_DIMS_SMALL(2),
    NHWC,
    COLOR_FORMATS_4CH);
}  // namespace smoke


// test everything in nightly (as before)
namespace nightly {
static auto params_csc_3ch_and_resize = COMBINE_WITH_DEFAULT(
    TESTED_DIMS(1),
    MULTI_VALUE(NCHW, NHWC),
    MULTI_VALUE(COLOR_FORMATS_RAW, COLOR_FORMATS_3CH));

static auto params_csc_4ch_and_resize = COMBINE_WITH_DEFAULT(
    TESTED_DIMS(1),
    NHWC,
    COLOR_FORMATS_4CH);

// batch preprocessing parameters:
static auto batch_params_csc_3ch_and_resize = COMBINE_WITH_DEFAULT(
    MULTI_VALUE(TESTED_DIMS(2), TESTED_DIMS(3)),
    MULTI_VALUE(NCHW, NHWC),
    MULTI_VALUE(COLOR_FORMATS_RAW, COLOR_FORMATS_3CH));

static auto batch_params_csc_4ch_and_resize = COMBINE_WITH_DEFAULT(
    MULTI_VALUE(TESTED_DIMS(2), TESTED_DIMS(3)),
    NHWC,
    COLOR_FORMATS_4CH);
}  // namespace nightly

// reorder preprocessing parameters:
static auto reorder_params = Combine(
        Values(Precision::FP32),  // network precision
        Values(SizeVector({1, 3, 300, 300})),  // sizes of the network
        Values(std::make_pair(Precision::FP32, 1e-2), std::make_pair(Precision::U8, 1)),  // precision and threshold
        Values(std::make_pair(NCHW, NHWC), std::make_pair(NHWC, NCHW)),  // Input/network data layout
        Values(ResizeAlgorithm::NO_RESIZE),
        Values(ColorFormat::BGR),
        Values(ROI({0, 0, 0, 300, 300})),  // cropped ROI params (id, x, y, width, height)
        Values(false, true)  // Infer mode sync/async
);

// nv12 preprocessing parameters:
static auto nv12_params = Combine(
        Values(Precision::FP32),  // network precision
        Values(cv::Size(300, 300)),  // input image size
        Values(TESTED_DIMS(1)),  // sizes of the network
        Values(std::make_pair(Precision::U8, 1)),  // precision and threshold
        Values(ResizeAlgorithm::RESIZE_BILINEAR, ResizeAlgorithm::RESIZE_AREA),
        Values(ColorFormat::NV12),
        Values(ROI({0, 0, 0, 300, 300}), ROI({0, 15, 10, 210, 210})),  // cropped ROI params (id, x, y, width, height)
        Values(false, true)  // Infer mode sync/async
);

static auto random_roi_3c = Combine(
            Values(Precision::FP32),
            Values(TESTED_DIMS(1)),
            Values(std::make_pair(Precision::FP32, 1e-2), std::make_pair(Precision::U8, 1)),
            Values(MULTI_VALUE(NCHW, NHWC)),
            Values(ResizeAlgorithm::RESIZE_BILINEAR, ResizeAlgorithm::RESIZE_AREA),
            Values(COLOR_FORMATS_3CH),
            Values(ROI({0, 0, 0, 0, 0})),
            Values(false, true)
);

static auto random_roi_4c = Combine(
            Values(Precision::FP32),
            Values(TESTED_DIMS(1)),
            Values(std::make_pair(Precision::FP32, 1e-2), std::make_pair(Precision::U8, 1)),
            Values(NHWC),
            Values(ResizeAlgorithm::RESIZE_BILINEAR, ResizeAlgorithm::RESIZE_AREA),
            Values(COLOR_FORMATS_4CH),
            Values(ROI({0, 0, 0, 0, 0})),
            Values(false, true)
);

static auto random_roi_nv12 = Combine(
            Values(Precision::FP32),
            Values(TESTED_DIMS(1)),
            Values(std::make_pair(Precision::U8, 1)),
            Values(NHWC),
            Values(ResizeAlgorithm::RESIZE_BILINEAR, ResizeAlgorithm::RESIZE_AREA),
            Values(ColorFormat::NV12),
            Values(ROI({0, 0, 0, 0, 0})),
            Values(false, true)
);
struct PreprocessRegression: public TestsCommon {};

TEST_F(PreprocessRegression, smoke_DifferentSizes) {
    // Reproduce "object was compiled for different meta" problem.
    // When G-API/Fluid is used as a preprocessing engine,
    // its state wasn't updated internally if input dimensions changed.
    // Thus while graph itself continued working properly on all dimensions,
    // it wan't reshaped when it had to:
    // * On first call (frame size = X), _lastCall is initialized with size X
    // * On second call (frame size = Y), graph is reshaped to size Y but _lastCall is still X
    // * On third call (frame size = X), graph is NOT reshaped since this X matches _lastCall,
    //   exception is thrown since a graph reshaped to input size Y is asked to process input size X.

    Blob::Ptr in_blob;
    Blob::Ptr out_blob;

    std::vector<cv::Size> in_sizes = {
        cv::Size(256, 256),
        cv::Size(72, 72),
        cv::Size(256, 256),
    };

    SizeVector out_dims = {1, 3, 64, 64};
    out_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, out_dims, Layout::NCHW));
    out_blob->allocate();

    PreProcessInfo info;
    info.setResizeAlgorithm(RESIZE_BILINEAR);

    PreProcessDataPtr preprocess = CreatePreprocDataHelper();
    for (auto sz : in_sizes) {
        cv::Mat in_mat = cv::Mat::eye(sz, CV_8UC3)*255;
        in_blob = img2Blob<Precision::U8>(in_mat, Layout::NHWC);
        preprocess->setRoiBlob(in_blob);
        EXPECT_NO_THROW(preprocess->execute(out_blob, info, false));
    }

    // Not thrown = test is green.
};

struct IEPreprocessTest : public TestsCommon {};
TEST_F(IEPreprocessTest, smoke_NetworkInputSmallSize) {
    const size_t num_threads = parallel_get_max_threads();

    std::vector<cv::Size> out_sizes = {
            cv::Size(num_threads, num_threads - 1),
            cv::Size(num_threads - 1, num_threads),
            cv::Size(1, 1),
            cv::Size(1, 0),
            cv::Size(0, 1)
    };

    SizeVector in_dims = {1, 3, num_threads * 2, num_threads * 2};
    cv::Mat in_mat = cv::Mat::eye(cv::Size(in_dims[3], in_dims[2]), CV_8UC3)*255;
    Blob::Ptr in_blob = img2Blob<Precision::U8>(in_mat, Layout::NHWC);

    PreProcessInfo info;
    info.setResizeAlgorithm(RESIZE_BILINEAR);

    PreProcessDataPtr preprocess = CreatePreprocDataHelper();
    preprocess->setRoiBlob(in_blob);

    for (const auto& sz : out_sizes) {
        SizeVector out_dims = {1, 3, static_cast<size_t>(sz.height), static_cast<size_t>(sz.width)};
        Blob::Ptr out_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, out_dims, Layout::NHWC));
        out_blob->allocate();
        // FIXME: sz with 0 dims must be a separate test
        if (sz.width > 0 && sz.height > 0) {
            EXPECT_NO_THROW(preprocess->execute(out_blob, info, false));
        } else {
            EXPECT_THROW(preprocess->execute(out_blob, info, false),
                         InferenceEngine::details::InferenceEngineException);
        }
    }
}

// smoke:
PLUGING_CASE_WITH_SUFFIX(CPU, _gapi_random_roi_3c_smoke, RandomROITest, random_roi_3c);
PLUGING_CASE_WITH_SUFFIX(CPU, _gapi_random_roi_4c_smoke, RandomROITest, random_roi_4c);
PLUGING_CASE_WITH_SUFFIX(CPU, _gapi_random_roi_nv12_smoke, RandomROITest, random_roi_nv12);

PLUGING_CASE_WITH_SUFFIX(CPU, _gapi_resize_only_smoke, CropResizeTest, smoke::params_resize_only);
PLUGING_CASE_WITH_SUFFIX(CPU, _gapi_csc_3ch_and_resize_smoke, CropResizeTest, smoke::params_csc_3ch_and_resize);
PLUGING_CASE_WITH_SUFFIX(CPU, _gapi_csc_4ch_and_resize_smoke, CropResizeTest, smoke::params_csc_4ch_and_resize);

PLUGING_CASE_WITH_SUFFIX(CPU, _gapi_resize_only_smoke, DynamicBatchResizeTest, smoke::batch_params_resize_only);
PLUGING_CASE_WITH_SUFFIX(CPU, _gapi_csc_3ch_and_resize_smoke, DynamicBatchResizeTest, smoke::batch_params_csc_3ch_and_resize);
PLUGING_CASE_WITH_SUFFIX(CPU, _gapi_csc_4ch_and_resize_smoke, DynamicBatchResizeTest, smoke::batch_params_csc_4ch_and_resize);

PLUGING_CASE_WITH_SUFFIX(CPU, _gapi_reorder_smoke, ReorderTest, reorder_params);

PLUGING_CASE_WITH_SUFFIX(CPU, _gapi_csc_nv12_and_resize_smoke, NV12ColorConvertTest, nv12_params);

////////////////////////////////////////////////////////////////////////////////////////////////////

// nightly:

// FIXME: enable these once smoke/nightly concepts are introduced in CI
PLUGING_CASE_WITH_SUFFIX(DISABLED_CPU, _gapi_random_roi_3c_nightly, RandomROITest, random_roi_3c);
PLUGING_CASE_WITH_SUFFIX(DISABLED_CPU, _gapi_random_roi_4c_nightly, RandomROITest, random_roi_4c);
PLUGING_CASE_WITH_SUFFIX(DISABLED_CPU, _gapi_random_roi_nv12_nightly, RandomROITest, random_roi_nv12);

PLUGING_CASE_WITH_SUFFIX(DISABLED_CPU, _gapi_csc_3ch_and_resize_nightly, CropResizeTest, nightly::params_csc_3ch_and_resize);
PLUGING_CASE_WITH_SUFFIX(DISABLED_CPU, _gapi_csc_4ch_and_resize_nightly, CropResizeTest, nightly::params_csc_4ch_and_resize);

PLUGING_CASE_WITH_SUFFIX(DISABLED_CPU, _gapi_csc_3ch_and_resize_nightly, BatchResizeTest, nightly::batch_params_csc_3ch_and_resize);
PLUGING_CASE_WITH_SUFFIX(DISABLED_CPU, _gapi_csc_4ch_and_resize_nightly, BatchResizeTest, nightly::batch_params_csc_4ch_and_resize);

PLUGING_CASE_WITH_SUFFIX(DISABLED_CPU, _gapi_csc_3ch_and_resize_nightly, DynamicBatchResizeTest, nightly::batch_params_csc_3ch_and_resize);
PLUGING_CASE_WITH_SUFFIX(DISABLED_CPU, _gapi_csc_4ch_and_resize_nightly, DynamicBatchResizeTest, nightly::batch_params_csc_4ch_and_resize);

PLUGING_CASE_WITH_SUFFIX(DISABLED_CPU, _gapi_reorder_nightly, ReorderTest, reorder_params);

PLUGING_CASE_WITH_SUFFIX(DISABLED_CPU, _gapi_csc_nv12_and_resize_nightly, NV12ColorConvertTest, nv12_params);

#endif  // USE_OPENCV
