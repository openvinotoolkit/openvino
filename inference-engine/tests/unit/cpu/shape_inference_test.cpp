// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/parameter.hpp>
#include <convolution_shape_inference.hpp>
#include <experimental_detectron_roi_feature_shape_inference.hpp>
#include <openvino/op/ops.hpp>
#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, ConvolutionTest) {
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto conv =
            std::make_shared<op::v1::Convolution>(data, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    std::vector<PartialShape> input_shapes = {PartialShape{3, 6, 5, 5}, PartialShape{7, 6, 3, 3}}, output_shapes = {PartialShape{}};
    shape_infer(conv.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], PartialShape({3, 7, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, StaticShape{7, 6, 3, 3}}, static_output_shapes = {StaticShape{}};
    shape_infer(conv.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 7, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

template <typename Callable>
static void perf_test(Callable func) {
    static int perf_test_N = std::getenv("PERFN") ? std::atoi(std::getenv("PERFN")) : 1;
    std::chrono::time_point<std::chrono::high_resolution_clock> before;
    std::chrono::time_point<std::chrono::high_resolution_clock> after;
    std::vector<std::chrono::nanoseconds::rep> diffs;

    if (perf_test_N == 1) {
        func();
        return;
    }

    for (size_t i = 0; i < perf_test_N; ++i) {
        before = std::chrono::high_resolution_clock::now();
        func();
        after = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        diffs.push_back(diff);
    }

    auto drop_percentage = 20;
    std::sort(diffs.begin(), diffs.end());
    auto skip = diffs.size() * drop_percentage / 2 / 100;
    auto sum = std::accumulate(diffs.begin() + skip, diffs.end() - skip, 0);
    auto avg = sum / (diffs.size() - 2 * skip);

    std::cout << " avg:" << avg << std::endl;
}

#define PERF_TEST(st) \
    perf_test([&]() { \
        st;           \
    })

TEST(StaticShapeInferenceTest, ExperimentalDetectronROIFeatureExtractor) {
    op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes attrs;
    attrs.aligned = false;
    attrs.output_size = 14;
    attrs.sampling_ratio = 2;
    attrs.pyramid_scales = {4, 8, 16, 32};

    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto pyramid_layer0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, -1, -1});
    auto pyramid_layer1 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, -1, -1});
    auto pyramid_layer2 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, -1, -1});
    auto pyramid_layer3 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, -1, -1});

    auto roi = std::make_shared<op::v6::ExperimentalDetectronROIFeatureExtractor>(
        NodeVector{input, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3},
        attrs);

    std::vector<PartialShape> input_shapes = {PartialShape{1000, 4},
                                              PartialShape{1, 256, 200, 336},
                                              PartialShape{1, 256, 100, 168},
                                              PartialShape{1, 256, 50, 84},
                                              PartialShape{1, 256, 25, 42}};
    std::vector<PartialShape> output_shapes = {PartialShape{}, PartialShape{}};
    PERF_TEST(shape_infer(roi.get(), input_shapes, output_shapes));

    ASSERT_EQ(roi->get_output_element_type(0), element::f32);

    EXPECT_EQ(output_shapes[0], (PartialShape{1000, 256, 14, 14}));
    EXPECT_EQ(output_shapes[1], (PartialShape{1000, 4}));

    std::vector<StaticShape> input_shapes2 = {StaticShape{1000, 4},
                                              StaticShape{1, 256, 200, 336},
                                              StaticShape{1, 256, 100, 168},
                                              StaticShape{1, 256, 50, 84},
                                              StaticShape{1, 256, 25, 42}};
    std::vector<StaticShape> output_shapes2 = {StaticShape{}, StaticShape{}};
    PERF_TEST(shape_infer(roi.get(), input_shapes2, output_shapes2));

    ASSERT_EQ(roi->get_output_element_type(0), element::f32);

    EXPECT_EQ(output_shapes2[0], (StaticShape{1000, 256, 14, 14}));
    EXPECT_EQ(output_shapes2[1], (StaticShape{1000, 4}));
}

#if 0
TEST(StaticShapeInferenceTest, ConvolutionTimeTest) {
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{3, 6, 5, 5});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{7, 6, 3, 3});
    auto conv =
            std::make_shared<op::v1::Convolution>(data, filters, strides, pads_begin, pads_end, dilations, auto_pad);
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, StaticShape{7, 6, 3, 3}}, static_output_shapes = {StaticShape{}};

    auto before = std::chrono::high_resolution_clock::now();
    auto after = std::chrono::high_resolution_clock::now();

    std::cout << conv << std::endl;
    auto convolution_time_sum = 0;
    for (size_t i = 0; i < 10; ++i) {
        before = std::chrono::high_resolution_clock::now();
        shape_infer(conv.get(), static_input_shapes, static_output_shapes);
        after = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        std::cout << diff << " ns" << std::endl;
        convolution_time_sum += diff;
    }

    // other operation creation and time measurements: ReLU is an example
    auto relu = std::make_shared<op::v0::Relu>(data);
    std::cout << relu << std::endl;
    auto other_op_time_sum = 0;
    for (size_t i = 0; i < 10; ++i) {
        before = std::chrono::high_resolution_clock::now();
        relu->validate_and_infer_types();
        after = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        std::cout << diff << " ns" << std::endl;
        other_op_time_sum += diff;
    }
    std::cout << (convolution_time_sum >= other_op_time_sum ? "ON PAR WITH CONVOLUTION: " : "LONGER THAN CONVOLUTION ")
              << 1. * other_op_time_sum / convolution_time_sum << std::endl;
}
#endif