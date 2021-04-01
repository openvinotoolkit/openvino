// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cropResize_tests.hpp"
#include "vpu_tests_config.hpp"

#ifdef USE_OPENCV

#define COMBINE_WITH_DEFAULT(_dims, _in_layouts, _color_formats) \
    Combine(Values(Precision::FP16), \
            Values(_dims), \
            Values(std::make_pair(Precision::FP32, 1e-1), std::make_pair(Precision::U8, 1)), \
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
        Values(Precision::FP16),  // network precision
        Values(SizeVector({1, 3, 300, 300})),  // sizes of the network
        Values(std::make_pair(Precision::FP32, 1e-1), std::make_pair(Precision::U8, 1)),  // precision and threshold
        Values(std::make_pair(NCHW, NHWC), std::make_pair(NHWC, NCHW)),  // Input/network data layout
        Values(ResizeAlgorithm::NO_RESIZE),
        Values(ColorFormat::BGR),
        Values(ROI({0, 0, 0, 300, 300})),  // cropped ROI params (id, x, y, width, height)
        Values(false, true)  // Infer mode sync/async
);

// nv12 preprocessing parameters:
static auto nv12_params = Combine(
        Values(Precision::FP16),  // network precision
        Values(cv::Size(300, 300)),  // input image size
        Values(TESTED_DIMS(1)),  // sizes of the network
        Values(std::make_pair(Precision::U8, 1)),  // precision and threshold
        Values(ResizeAlgorithm::RESIZE_BILINEAR, ResizeAlgorithm::RESIZE_AREA),
        Values(ColorFormat::NV12),
        Values(ROI({0, 0, 0, 300, 300}), ROI({0, 15, 10, 210, 210})),  // cropped ROI params (id, x, y, width, height)
        Values(false, true)  // Infer mode sync/async
);

static auto random_roi_3c = Combine(
            Values(Precision::FP16),
            Values(TESTED_DIMS(1)),
            Values(std::make_pair(Precision::FP32, 1e-2), std::make_pair(Precision::U8, 1)),
            Values(MULTI_VALUE(NCHW, NHWC)),
            Values(ResizeAlgorithm::RESIZE_BILINEAR, ResizeAlgorithm::RESIZE_AREA),
            Values(COLOR_FORMATS_3CH),
            Values(ROI({0, 0, 0, 0, 0})),
            Values(false, true)
);

static auto random_roi_4c = Combine(
            Values(Precision::FP16),
            Values(TESTED_DIMS(1)),
            Values(std::make_pair(Precision::FP32, 1e-2), std::make_pair(Precision::U8, 1)),
            Values(NHWC),
            Values(ResizeAlgorithm::RESIZE_BILINEAR, ResizeAlgorithm::RESIZE_AREA),
            Values(COLOR_FORMATS_4CH),
            Values(ROI({0, 0, 0, 0, 0})),
            Values(false, true)
);

static auto random_roi_nv12 = Combine(
            Values(Precision::FP16),
            Values(TESTED_DIMS(1)),
            Values(std::make_pair(Precision::U8, 1)),
            Values(NHWC),
            Values(ResizeAlgorithm::RESIZE_BILINEAR, ResizeAlgorithm::RESIZE_AREA),
            Values(ColorFormat::NV12),
            Values(ROI({0, 0, 0, 0, 0})),
            Values(false, true)
);

// smoke:
VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_random_roi_c3_smoke, RandomROITest, random_roi_3c);
VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_random_roi_c4_smoke, RandomROITest, random_roi_4c);
VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_random_roi_nv12_smoke, RandomROITest, random_roi_nv12);

VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_resize_only_smoke, CropResizeTest, smoke::params_resize_only);
VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_csc_3ch_and_resize_smoke, CropResizeTest, smoke::params_csc_3ch_and_resize);
VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_csc_4ch_and_resize_smoke, CropResizeTest, smoke::params_csc_4ch_and_resize);

VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_resize_only_smoke, BatchResizeTest, smoke::batch_params_resize_only);
VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_csc_3ch_and_resize_smoke, BatchResizeTest, smoke::batch_params_csc_3ch_and_resize);
VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_csc_4ch_and_resize_smoke, BatchResizeTest, smoke::batch_params_csc_4ch_and_resize);

VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_reorder_smoke, ReorderTest, reorder_params);

VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_csc_nv12_and_resize_smoke, NV12ColorConvertTest, nv12_params);

////////////////////////////////////////////////////////////////////////////////////////////////////

// nightly:

// FIXME: enable these once smoke/nightly concepts are introduced in CI
DISABLED_VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_random_roi_c3_nightly, RandomROITest, random_roi_3c);
DISABLED_VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_random_roi_c4_nightly, RandomROITest, random_roi_4c);
DISABLED_VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_random_roi_nv12_nightly, RandomROITest, random_roi_nv12);

DISABLED_VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_csc_3ch_and_resize_nightly, CropResizeTest, nightly::params_csc_3ch_and_resize);
DISABLED_VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_csc_4ch_and_resize_nightly, CropResizeTest, nightly::params_csc_4ch_and_resize);

DISABLED_VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_csc_3ch_and_resize_nightly, BatchResizeTest, nightly::batch_params_csc_3ch_and_resize);
DISABLED_VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_csc_4ch_and_resize_nightly, BatchResizeTest, nightly::batch_params_csc_4ch_and_resize);

DISABLED_VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_reorder_nightly, ReorderTest, reorder_params);

DISABLED_VPU_PLUGING_CASE_WITH_SUFFIX(_gapi_csc_nv12_and_resize_nightly, NV12ColorConvertTest, nv12_params);

#endif  // USE_OPENCV
