// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"

#include <vector>

using namespace ov::test;

namespace {

using namespace ov::builder::preprocess;

inline std::vector<preprocess_func> GPU_smoke_preprocess_functions() {
    return std::vector<preprocess_func>{
        preprocess_func(mean_only, "mean_only", 0.01f),
        preprocess_func(scale_only, "scale_only", 0.01f),
        preprocess_func(mean_scale, "mean_scale", 0.01f),
        preprocess_func(scale_mean, "scale_mean", 0.01f),
        preprocess_func(mean_vector, "mean_vector", 0.01f),
        preprocess_func(scale_vector, "scale_vector", 0.01f),
        preprocess_func(two_inputs_basic, "two_inputs_basic", 0.01f),
        preprocess_func(two_inputs_trivial, "two_inputs_trivial", 0.01f),
        preprocess_func(reuse_network_layout, "reuse_network_layout", 0.01f),
        preprocess_func(tensor_layout, "tensor_layout", 0.01f),
        preprocess_func(resize_linear, "resize_linear", 0.01f),
        preprocess_func(resize_nearest, "resize_nearest", 0.01f),
        preprocess_func(resize_linear_nhwc, "resize_linear_nhwc", 0.01f),
        preprocess_func(resize_cubic, "resize_cubic", 0.01f),
        preprocess_func(resize_dynamic, "resize_dynamic", 0.01f, {ov::Shape{1, 3, 123, 123}}),
        preprocess_func(crop_basic, "crop_basic", 0.000001f),
        preprocess_func(crop_negative, "crop_negative", 0.000001f),
        preprocess_func(convert_layout_by_dims, "convert_layout_by_dims", 0.01f),
        preprocess_func(convert_layout_hwc_to_nchw, "convert_layout_hwc_to_nchw", 0.01f),
        preprocess_func(resize_and_convert_layout, "resize_and_convert_layout", 0.01f),
        preprocess_func(cvt_color_nv12_to_rgb_single_plane, "cvt_color_nv12_to_rgb_single_plane", 1.f),
        preprocess_func(cvt_color_nv12_to_bgr_two_planes, "cvt_color_nv12_to_bgr_two_planes", 1.f),
        preprocess_func(cvt_color_nv12_cvt_layout_resize, "cvt_color_nv12_cvt_layout_resize", 1.f),
        preprocess_func(cvt_color_i420_to_rgb_single_plane, "cvt_color_i420_to_rgb_single_plane", 1.f),
        preprocess_func(cvt_color_i420_to_bgr_three_planes, "cvt_color_i420_to_bgr_three_planes", 1.f),
        preprocess_func(cvt_color_bgrx_to_bgr, "cvt_color_bgrx_to_bgr", 0.01f),
    };
}

INSTANTIATE_TEST_SUITE_P(smoke_PrePostProcess_GPU,
                         PrePostProcessTest,
                         ::testing::Combine(::testing::ValuesIn(GPU_smoke_preprocess_functions()),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         PrePostProcessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_PostProcess,
    PostProcessTest,
    ::testing::Combine(::testing::ValuesIn(ov::builder::preprocess::generic_postprocess_functions()),
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    PostProcessTest::getTestCaseName);

}  // namespace
