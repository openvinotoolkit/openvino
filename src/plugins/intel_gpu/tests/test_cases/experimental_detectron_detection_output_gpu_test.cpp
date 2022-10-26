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

    size_t roi_count;

    std::vector<T> boxes;
    std::vector<T> deltas;
    std::vector<T> scores;
    std::vector<T> im_info;

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
            engine.allocate_memory({data_type, format::bfyx, tensor{batch(param.roi_count), feature(4)}});
        set_values(input_boxes, param.boxes);

        const primitive_id input_deltas_id = "InputDeltas";
        auto input_deltas = engine.allocate_memory(
            {data_type, format::bfyx, tensor{batch(param.roi_count), feature(param.num_classes * 4)}});
        set_values(input_deltas, param.deltas);

        const primitive_id input_scores_id = "InputScores";
        auto input_scores = engine.allocate_memory(
            {data_type, format::bfyx, tensor{batch(param.roi_count), feature(param.num_classes)}});
        set_values(input_scores, param.scores);

        const primitive_id input_im_info_id = "InputImInfo";
        const auto input_im_info = engine.allocate_memory({data_type, format::bfyx, tensor{batch(1), feature(3)}});
        set_values(input_im_info, param.im_info);

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
        for (int i = 0; i < param.max_detections_per_image; ++i) {
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
            0.01000000074505806f,       // score_threshold
            0.2f,                       // nms_threshold
            2.0f,                       // max_delta_log_wh
            2,                          // num_classes
            500,                        // post_nms_count
            5,                          // max_detections_per_image
            true,                       // class_agnostic_box_regression
            {10.0f, 10.0f, 5.0f, 5.0f}, // deltas_weights
            16,                         // roi count

            // boxes
            getValues<T>({1.0f, 1.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 8.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}),

            // deltas
            getValues<T>(
                {5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                 1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}),

            // scores
            getValues<T>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}),

            // im_info
            getValues<T>({1.0f, 1.0f, 1.0f}),

            // out_boxes
            getValues<T>({0.8929862f,
                          0.892986297607421875,
                          12.10701370239257812,
                          12.10701370239257812,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0}),

            // out_classes
            std::vector<int32_t>{1, 0, 0, 0, 0},

            // out_scores
            getValues<T>({1.0f, 0.0f, 0.0f, 0.0f, 0.0f})
        },
        {
            0.01000000074505806f,        // score_threshold
            0.2f,                        // nms_threshold
            2.0f,                        // max_delta_log_wh
            2,                           // num_classes
            500,                         // post_nms_count
            5,                           // max_detections_per_image
            true,                        // class_agnostic_box_regression
            {10.0f, 10.0f, 5.0f, 5.0f},  // deltas_weights
            16,                          // roi count

            // boxes
            getValues<T>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f,
                          1.0f, 8.0f, 5.0f, 1.0f, 1.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f}),

            // deltas
            getValues<T>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  1.0f}),

            // scores
            getValues<T>({0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.8f, 0.9f, 0.5f,
                          0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                          0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}),

            // im_info
            getValues<T>({16.0f, 12.0f, 1.0f}),

            // out_boxes
            getValues<T>({ 0.0f, 0.892987f, 10.107f, 12.107f, 0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f,       0.0f,    0.0f,   0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f,       0.0f,    0.0f}),

            // out_classes
            std::vector<int32_t>{1, 0, 0, 0, 0},

            // out_scores
            getValues<T>({0.9f, 0.0f, 0.0f, 0.0f, 0.0f}),
        },
        {
            0.0500000007,                // score_threshold
            0.5,                         // nms_threshold
            4.13516665,                  // max_delta_log_wh
            5,                           // num_classes
            10,                          // post_nms_count
            10,                          // max_detections_per_image
            false,                       // class_agnostic_box_regression
            {10.0f, 10.0f, 5.0f, 5.0f},  // deltas_weights
            10,                          // roi count

            // boxes
            getValues<T>({
                4.90234,  6.57812, 5.23828,  9.19531, 8.51172, 2,       8.22266,  0.492188, 9.87109,  4.17188,
                6.95703,  8.53906, 0.980469, 9.09375, 3.44141, 5.33594, 9.83984,  6.76562,  1.67578,  6.88281,
                0.449219, 9.1875,  7.66016,  7.17969, 8.80859, 2.35938, 5.39453,  8.22656,  0.917969, 0.28125,
                6.87891,  6.02344, 6.77734,  6.95312, 6.11328, 6.57031, 0.386719, 8.375,    5.09766,  9.86719,
            }),

            // deltas
            getValues<T>({
                4.90234,  6.57812,  5.23828,  9.19531,   8.51172,  2,        8.22266,   0.492188, 9.87109,  4.17188,
                6.95703,  8.53906,  0.980469, 9.09375,   3.44141,  5.33594,  9.83984,   6.76562,  1.67578,  6.88281,
                0.449219, 9.1875,   7.66016,  7.17969,   8.80859,  2.35938,  5.39453,   8.22656,  0.917969, 0.28125,
                6.87891,  6.02344,  6.77734,  6.95312,   6.11328,  6.57031,  0.386719,  8.375,    5.09766,  9.86719,
                3.74609,  4.54688,  5.83203,  5.91406,   2.85547,  7.46875,  4.31641,   2.71094,  9.71484,  1.14062,
                6.55078,  0.257812, 4.32422,  9.5625,    8.53516,  0.554688, 8.68359,   2.73438,  6.26953,  5.60156,
                2.79297,  8.65625,  5.75391,  5.39844,   2.65234,  7.32812,  8.98828,   7.94531,  6.26172,  4.75,
                7.97266,  1.24219,  5.62109,  8.92188,   2.70703,  1.28906,  4.73047,   7.84375,  5.19141,  6.08594,
                7.58984,  9.51562,  7.42578,  5.63281,   6.19922,  7.9375,   5.41016,   9.92969,  2.55859,  1.10938,
                1.14453,  8.97656,  4.66797,  9.03125,   4.62891,  0.773438, 4.52734,   1.70312,  9.86328,  1.32031,
                0.136719, 9.125,    2.84766,  4.61719,   9.49609,  5.29688,  5.58203,   0.664062, 2.60547,  6.21875,
                8.06641,  5.46094,  1.46484,  7.89062,   0.300781, 5.00781,  0.0742188, 0.3125,   6.28516,  3.30469,
                4.43359,  1.48438,  2.01953,  8.35156,   8.54297,  7.40625,  9.50391,   2.14844,  2.40234,  2.07812,
                2.73828,  2.69531,  4.01172,  9.5,       7.72266,  9.99219,  1.37109,   3.67188,  2.45703,  2.03906,
                0.480469, 4.59375,  2.94141,  4.83594,   1.33984,  0.265625, 1.17578,   4.38281,  5.94922,  8.6875,
                5.16016,  0.679688, 4.30859,  5.85938,   4.89453,  7.72656,  4.41797,   5.78125,  4.37891,  1.52344,
                8.27734,  4.45312,  3.61328,  4.07031,   7.88672,  9.875,    4.59766,   1.36719,  7.24609,  8.04688,
                5.33203,  5.41406,  4.35547,  0.96875,   1.81641,  8.21094,  3.21484,   4.64062,  4.05078,  9.75781,
                7.82422,  3.0625,   4.03516,  0.0546875, 8.18359,  8.23438,  1.76953,   1.10156,  2.29297,  8.15625,
                9.25391,  0.898438, 6.15234,  8.82812,   6.48828,  7.44531,  1.76172,   2.25,     9.47266,  0.742188,
            }),

            // scores
            getValues<T>({
                4.90234,  6.57812, 5.23828,  9.19531, 8.51172, 2,       8.22266,  0.492188, 9.87109,  4.17188,
                6.95703,  8.53906, 0.980469, 9.09375, 3.44141, 5.33594, 9.83984,  6.76562,  1.67578,  6.88281,
                0.449219, 9.1875,  7.66016,  7.17969, 8.80859, 2.35938, 5.39453,  8.22656,  0.917969, 0.28125,
                6.87891,  6.02344, 6.77734,  6.95312, 6.11328, 6.57031, 0.386719, 8.375,    5.09766,  9.86719,
                3.74609,  4.54688, 5.83203,  5.91406, 2.85547, 7.46875, 4.31641,  2.71094,  9.71484,  1.14062,
            }),

            // im_info
            getValues<T>({
                4.90234,
                6.57812,
                5.23828,
            }),

            // out_boxes
            getValues<T>({ 0.0f,     2.97829f, 14.8295f,  11.1221f, 0.0f,     6.29737f, 16.2088f,  16.3451f,
                           4.37184f, 6.41816f,  6.03075f, 15.934f,  5.95092f, 3.66966f,  6.81878f, 16.9983f,
                           0.0f,     5.64766f, 17.3085f,  12.3716f, 1.31074f, 9.12453f, 13.1104f,  10.6441f,
                           3.24828f, 7.11447f,  9.16656f, 10.1058f, 0.0f,     0.0f,     10.0008f,  14.6173f,
                           4.20346f, 0.0f,      8.5746f,  18.8736f, 0.0f,     0.0f,     15.661f,   22.4114f}
            ),

            // out_classes
            std::vector<int32_t>({
                4,
                3,
                3,
                4,
                2,
                0,
                1,
                0,
                2,
                3,
            }),

            // out_scores
            getValues<T>({
                9.86719,
                9.71484,
                9.19531,
                8.51172,
                8.375,
                7.46875,
                6.57812,
                6.57031,
                5.23828,
                5.09766,
            }),
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
