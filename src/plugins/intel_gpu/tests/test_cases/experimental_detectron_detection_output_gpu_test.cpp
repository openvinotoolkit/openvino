// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/experimental_detectron_detection_output.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {
constexpr size_t roi_count = 16;

const std::vector<float> boxes{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f,
                               1.0f, 8.0f, 5.0f, 1.0f, 1.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

const std::vector<float> deltas{
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

const std::vector<float> scores{0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.8f, 0.9f, 0.5f,
                                0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                                0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

const std::vector<float> im_info{16.0f, 12.0f, 1.0f};

template <typename T>
std::vector<T> getValues(const std::vector<float>& values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template <typename T>
float getError();

template <>
float getError<float>() {
    return 0.001;
}

template <>
float getError<half_t>() {
    return 0.2;
}

};  // namespace

template <typename T>
struct ExperimentalDetectronDetectionOutputParams {
    float score_threshold;
    float nms_threshold;
    float max_delta_log_wh;
    int num_classes;
    int post_nms_count;
    int max_detections_per_image;
    bool class_agnostic_box_regression;
    std::vector<float> deltas_weights;

    std::vector<T> expected_boxes;
    std::vector<int32_t> expected_classes;
    std::vector<T> expected_scores;
};

template <typename T>
struct experimental_detectron_detection_output_test
    : public ::testing::TestWithParam<ExperimentalDetectronDetectionOutputParams<T>> {
public:
    void test() {
        const ExperimentalDetectronDetectionOutputParams<T> param =
            testing::TestWithParam<ExperimentalDetectronDetectionOutputParams<T>>::GetParam();
        auto data_type = type_to_data_type<T>::value;

        auto& engine = get_test_engine();

        const primitive_id input_boxes_id = "InputBoxes";
        const auto input_boxes =
            engine.allocate_memory({data_type, format::bfyx, tensor{batch(roi_count), feature(4)}});
        set_values(input_boxes, getValues<T>(boxes));

        const primitive_id input_deltas_id = "InputDeltas";
        auto input_deltas =
            engine.allocate_memory({data_type, format::bfyx, tensor{batch(roi_count), feature(param.num_classes * 4)}});
        set_values(input_deltas, getValues<T>(deltas));

        const primitive_id input_scores_id = "InputScores";
        auto input_scores =
            engine.allocate_memory({data_type, format::bfyx, tensor{batch(roi_count), feature(param.num_classes)}});
        set_values(input_scores, getValues<T>(scores));

        const primitive_id input_im_info_id = "InputImInfo";
        const auto input_im_info = engine.allocate_memory({data_type, format::bfyx, tensor{batch(1), feature(3)}});
        set_values(input_im_info, getValues<T>(im_info));

        const primitive_id output_scores_id = "OutputScores";
        auto output_scores =
            engine.allocate_memory({data_type, format::bfyx, tensor{batch(param.max_detections_per_image)}});

        const primitive_id output_classes_id = "OutputClasses";
        auto output_classes =
            engine.allocate_memory({data_types::i32, format::bfyx, tensor{batch(param.max_detections_per_image)}});

        topology topology;

        topology.add(input_layout(input_boxes_id, input_boxes->get_layout()));
        topology.add(input_layout(input_deltas_id, input_deltas->get_layout()));
        topology.add(input_layout(input_scores_id, input_scores->get_layout()));
        topology.add(input_layout(input_im_info_id, input_im_info->get_layout()));
        topology.add(mutable_data(output_classes_id, output_classes));
        topology.add(mutable_data(output_scores_id, output_scores));

        const primitive_id eddo_id = "experimental_detectron_detection_output";
        const auto eddo_primitive = experimental_detectron_detection_output{
            eddo_id,
            input_boxes_id,
            input_deltas_id,
            input_scores_id,
            input_im_info_id,
            output_classes_id,
            output_scores_id,
            param.score_threshold,
            param.nms_threshold,
            param.num_classes,
            param.post_nms_count,
            param.max_detections_per_image,
            param.class_agnostic_box_regression,
            param.max_delta_log_wh,
            param.deltas_weights,
        };

        topology.add(eddo_primitive);

        network network(engine, topology);

        network.set_input_data(input_boxes_id, input_boxes);
        network.set_input_data(input_deltas_id, input_deltas);
        network.set_input_data(input_scores_id, input_scores);
        network.set_input_data(input_im_info_id, input_im_info);

        const auto outputs = network.execute();

        const auto output_boxes = outputs.at(eddo_id).get_memory();

        const cldnn::mem_lock<T> output_boxes_ptr(output_boxes, get_test_stream());
        ASSERT_EQ(output_boxes_ptr.size(), param.max_detections_per_image * 4);

        const cldnn::mem_lock<int32_t> output_classes_ptr(output_classes, get_test_stream());
        ASSERT_EQ(output_classes_ptr.size(), param.max_detections_per_image);

        const cldnn::mem_lock<T> output_scores_ptr(output_scores, get_test_stream());
        ASSERT_EQ(output_scores_ptr.size(), param.max_detections_per_image);

        const auto& expected_boxes = param.expected_boxes;
        const auto& expected_classes = param.expected_classes;
        const auto& expected_scores = param.expected_scores;
        for (size_t i = 0; i < param.max_detections_per_image; ++i) {
            EXPECT_NEAR(expected_scores[i], output_scores_ptr[i], 0.001) << "i=" << i;
            for (size_t coord = 0; coord < 4; ++coord) {
                const auto roi_idx = i * 4 + coord;
                EXPECT_NEAR(expected_boxes[roi_idx], output_boxes_ptr[roi_idx], getError<T>())
                    << "i=" << i << ", coord=" << coord;
            }

            EXPECT_EQ(expected_classes[i], output_classes_ptr[i]) << "i=" << i;
        }
    }
};

using experimental_detectron_detection_output_test_f32 = experimental_detectron_detection_output_test<float>;
using experimental_detectron_detection_output_test_f16 = experimental_detectron_detection_output_test<half_t>;

TEST_P(experimental_detectron_detection_output_test_f32, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(experimental_detectron_detection_output_test_f16, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}

template <typename T>
std::vector<ExperimentalDetectronDetectionOutputParams<T>> getExperimentalDetectronDetectionOutputParams() {
    std::vector<ExperimentalDetectronDetectionOutputParams<T>> params = {
        {
            0.01000000074505806f,        // score_threshold
            0.2f,                        // nms_threshold
            2.0f,                        // max_delta_log_wh
            2,                           // num_classes
            500,                         // post_nms_count
            5,                           // max_detections_per_image
            true,                        // class_agnostic_box_regression
            {10.0f, 10.0f, 5.0f, 5.0f},  // deltas_weights
            getValues<T>({4.8929863f,  0.892986298f, 12.0f, 12.1070137f, 0.0f, 0.892986298f, 10.1070137f,
                          12.1070137f, 0.0f,         0.0f,  0.0f,        0.0f, 0.0f,         0.0f,
                          0.0f,        0.0f,         0.0f,  0.0f,        0.0f, 0.0f}),
            std::vector<int32_t>{0, 1, 0, 0, 0},
            getValues<T>({0.8f, 0.9f, 0.0f, 0.0f, 0.0f}),
        },
    };
    return params;
}

INSTANTIATE_TEST_SUITE_P(experimental_detectron_detection_output_gpu_test,
                         experimental_detectron_detection_output_test_f32,
                         ::testing::ValuesIn(getExperimentalDetectronDetectionOutputParams<float>()));

INSTANTIATE_TEST_SUITE_P(experimental_detectron_detection_output_gpu_test,
                         experimental_detectron_detection_output_test_f16,
                         ::testing::ValuesIn(getExperimentalDetectronDetectionOutputParams<half_t>()));
