// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/generate_proposals.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>

using namespace cldnn;
using namespace ::tests;

namespace {
template <typename T>
struct GenerateProposalsParams {
    float min_size;
    float nms_threshold;
    int64_t pre_nms_count;
    int64_t post_nms_count;
    bool normalized;
    float nms_eta;
    std::vector<T> expected_rois;
    std::vector<T> expected_roi_scores;
    std::vector<size_t> expected_rois_num;
};

template <typename T>
using GenerateProposalsParamsWithLayout = std::tuple<GenerateProposalsParams<T>, format::type>;

constexpr size_t num_batches = 2;
constexpr size_t image_height = 200;
constexpr size_t image_width = 200;
constexpr float image_scale = 4.0f;
constexpr size_t height = 2;
constexpr size_t width = 6;
constexpr size_t number_of_channels = 3;
constexpr size_t number_of_anchors = 3;

const std::vector<float> im_info{image_height, image_width, image_scale, image_height, image_width, image_scale};

const std::vector<float> anchors{
        0.0f,  1.0f,  2.0f,  3.0f,
        4.0f,  5.0f,  6.0f,  7.0f,
        8.0f,  9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f, 27.0f,
        28.0f, 29.0f, 30.0f, 31.0f,
        32.0f, 33.0f, 34.0f, 35.0f,
        36.0f, 37.0f, 38.0f, 39.0f,
        40.0f, 41.0f, 42.0f, 43.0f,
        44.0f, 45.0f, 46.0f, 47.0f,
        48.0f, 49.0f, 50.0f, 51.0f,
        52.0f, 53.0f, 54.0f, 55.0f,
        56.0f, 57.0f, 58.0f, 59.0f,
        60.0f, 61.0f, 62.0f, 63.0f,
        64.0f, 65.0f, 66.0f, 67.0f,
        68.0f, 69.0f, 70.0f, 71.0f,
        72.0f, 73.0f, 74.0f, 75.0f,
        76.0f, 77.0f, 78.0f, 79.0f,
        80.0f, 81.0f, 82.0f, 83.0f,
        84.0f, 85.0f, 86.0f, 87.0f,
        88.0f, 89.0f, 90.0f, 91.0f,
        92.0f, 93.0f, 94.0f, 95.0f,
        96.0f,  97.0f,  98.0f,  99.0f,
        100.0f, 101.0f, 102.0f, 103.0f,
        104.0f, 105.0f, 106.0f, 107.0f,
        108.0f, 109.0f, 110.0f, 111.0f,
        112.0f, 113.0f, 114.0f, 115.0f,
        116.0f, 117.0f, 118.0f, 119.0f,
        120.0f, 121.0f, 122.0f, 123.0f,
        124.0f, 125.0f, 126.0f, 127.0f,
        128.0f, 129.0f, 130.0f, 131.0f,
        132.0f, 133.0f, 134.0f, 135.0f,
        136.0f, 137.0f, 138.0f, 139.0f,
        140.0f, 141.0f, 142.0f, 143.0f};

const std::vector<float> deltas{
        0.5337073,  0.86607957, 0.55151343, 0.21626699, 0.4462629,  0.03985678,
        0.5157072,  0.9932138,  0.7565954,  0.43803605, 0.802818,   0.14834064,
        0.53932905, 0.14314,    0.3817048,  0.95075196, 0.05516243, 0.2567484,
        0.25508744, 0.77438325, 0.43561,    0.2094628,  0.8299043,  0.44982538,
        0.95615596, 0.5651084,  0.11801951, 0.05352486, 0.9774733,  0.14439464,
        0.62644225, 0.14370479, 0.54161614, 0.557915,   0.53102225, 0.0840179,
        0.7249888,  0.9843559,  0.5490522,  0.53788143, 0.822474,   0.3278008,
        0.39688024, 0.3286012,  0.5117038,  0.04743988, 0.9408995,  0.29885054,
        0.81039643, 0.85277915, 0.06807619, 0.86430097, 0.36225632, 0.16606331,
        0.5401001,  0.7541649,  0.11998601, 0.5131829,  0.40606487, 0.327888,
        0.27721855, 0.6378373,  0.22795396, 0.4961256,  0.3215895,  0.15607187,
        0.14782153, 0.8908137,  0.8835288,  0.834191,   0.29907143, 0.7983525,
        0.755875,   0.30837986, 0.0839176,  0.26624718, 0.04371626, 0.09472824,
        0.20689541, 0.37622106, 0.1083321,  0.1342548,  0.05815459, 0.7676379,
        0.8105144,  0.92348766, 0.26761323, 0.7183306,  0.8947588,  0.19020908,
        0.42731014, 0.7473663,  0.85775334, 0.9340091,  0.3278848,  0.755993,
        0.05307213, 0.39705503, 0.21003333, 0.5625373,  0.66188884, 0.80521655,
        0.6125863,  0.44678232, 0.97802377, 0.0204936,  0.02686367, 0.7390654,
        0.74631,    0.58399844, 0.5988792,  0.37413648, 0.5946692,  0.6955776,
        0.36377597, 0.7891322,  0.40900692, 0.99139464, 0.50169915, 0.41435778,
        0.17142445, 0.26761186, 0.31591868, 0.14249913, 0.12919712, 0.5418711,
        0.6523203,  0.50259084, 0.7379765,  0.01171071, 0.94423133, 0.00841132,
        0.97486794, 0.2921785,  0.7633071,  0.88477814, 0.03563205, 0.50833166,
        0.01354555, 0.535081,   0.41366324, 0.0694767,  0.9944055,  0.9981207,
        0.5337073,  0.86607957, 0.55151343, 0.21626699, 0.4462629,  0.03985678,
        0.5157072,  0.9932138,  0.7565954,  0.43803605, 0.802818,   0.14834064,
        0.53932905, 0.14314,    0.3817048,  0.95075196, 0.05516243, 0.2567484,
        0.25508744, 0.77438325, 0.43561,    0.2094628,  0.8299043,  0.44982538,
        0.95615596, 0.5651084,  0.11801951, 0.05352486, 0.9774733,  0.14439464,
        0.62644225, 0.14370479, 0.54161614, 0.557915,   0.53102225, 0.0840179,
        0.7249888,  0.9843559,  0.5490522,  0.53788143, 0.822474,   0.3278008,
        0.39688024, 0.3286012,  0.5117038,  0.04743988, 0.9408995,  0.29885054,
        0.81039643, 0.85277915, 0.06807619, 0.86430097, 0.36225632, 0.16606331,
        0.5401001,  0.7541649,  0.11998601, 0.5131829,  0.40606487, 0.327888,
        0.27721855, 0.6378373,  0.22795396, 0.4961256,  0.3215895,  0.15607187,
        0.14782153, 0.8908137,  0.8835288,  0.834191,   0.29907143, 0.7983525,
        0.755875,   0.30837986, 0.0839176,  0.26624718, 0.04371626, 0.09472824,
        0.20689541, 0.37622106, 0.1083321,  0.1342548,  0.05815459, 0.7676379,
        0.8105144,  0.92348766, 0.26761323, 0.7183306,  0.8947588,  0.19020908,
        0.42731014, 0.7473663,  0.85775334, 0.9340091,  0.3278848,  0.755993,
        0.05307213, 0.39705503, 0.21003333, 0.5625373,  0.66188884, 0.80521655,
        0.6125863,  0.44678232, 0.97802377, 0.0204936,  0.02686367, 0.7390654,
        0.74631,    0.58399844, 0.5988792,  0.37413648, 0.5946692,  0.6955776,
        0.36377597, 0.7891322,  0.40900692, 0.99139464, 0.50169915, 0.41435778,
        0.17142445, 0.26761186, 0.31591868, 0.14249913, 0.12919712, 0.5418711,
        0.6523203,  0.50259084, 0.7379765,  0.01171071, 0.94423133, 0.00841132,
        0.97486794, 0.2921785,  0.7633071,  0.88477814, 0.03563205, 0.50833166,
        0.01354555, 0.535081,   0.41366324, 0.0694767,  0.9944055,  0.9981207};

const std::vector<float> scores{
        0.56637216, 0.90457034, 0.69827306, 0.4353543,  0.47985056, 0.42658508,
        0.14516132, 0.08081771, 0.1799732,  0.9229515,  0.42420176, 0.50857586,
        0.82664067, 0.4972319,  0.3752427,  0.56731623, 0.18241242, 0.33252355,
        0.30608943, 0.6572437,  0.69185436, 0.88646156, 0.36985755, 0.5590753,
        0.5256446,  0.03342898, 0.1344396,  0.68642473, 0.37953874, 0.32575172,
        0.21108444, 0.5661886,  0.45378175, 0.62126315, 0.26799858, 0.37272978,
        0.56637216, 0.90457034, 0.69827306, 0.4353543,  0.47985056, 0.42658508,
        0.14516132, 0.08081771, 0.1799732,  0.9229515,  0.42420176, 0.50857586,
        0.82664067, 0.4972319,  0.3752427,  0.56731623, 0.18241242, 0.33252355,
        0.30608943, 0.6572437,  0.69185436, 0.88646156, 0.36985755, 0.5590753,
        0.5256446,  0.03342898, 0.1344396,  0.68642473, 0.37953874, 0.32575172,
        0.21108444, 0.5661886,  0.45378175, 0.62126315, 0.26799858, 0.37272978};

const std::vector<format::type> layouts{
    format::bfyx,
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv16,
    format::bs_fs_yx_bsv32_fsv32};

template <typename T>
std::vector<T> getValues(const std::vector<float>& values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}
template <typename T> float getError();

template<>
float getError<float>() {
    return 0.001;
}

template<>
float getError<half_t>() {
    return 0.2;
}

template <typename T>
std::vector<GenerateProposalsParams<T>> getGenerateProposalsParams() {
    std::vector<GenerateProposalsParams<T>> params = {
            {
                    1.0f, 0.7f, 14, 6, true, 1.0,
                    getValues<T>({4.49132, 4.30537, 8.75027, 8.8035,
                                  0, 1.01395, 4.66909, 5.14337,
                                  135.501, 137.467, 139.81, 141.726,
                                  4.49132, 4.30537, 8.75027, 8.8035,
                                  0, 1.01395, 4.66909, 5.14337,
                                  135.501, 137.467, 139.81, 141.726}),
                    getValues<T>({0.826641, 0.566372, 0.559075,
                                  0.826641, 0.566372, 0.559075}),
                    {3, 3}
            },
            {
                    1.0f, 0.7f, 1000, 6, true, 1.0,
                    getValues<T>({4.49132, 4.30537, 8.75027, 8.8035,
                                  0, 1.01395, 4.66909, 5.14337,
                                  135.501, 137.467, 139.81, 141.726,
                                  47.2348, 47.8342, 52.5503, 52.3864,
                                  126.483, 128.3, 131.625, 133.707,
                                  4.49132, 4.30537, 8.75027, 8.8035,
                                  0, 1.01395, 4.66909, 5.14337,
                                  135.501, 137.467, 139.81, 141.726,
                                  47.2348, 47.8342, 52.5503, 52.3864,
                                  126.483, 128.3, 131.625, 133.707}),
                    getValues<T>({0.826641, 0.566372, 0.559075, 0.479851, 0.267999,
                                  0.826641, 0.566372, 0.559075, 0.479851, 0.267999}),
                    {5, 5}
            },
            {
                    0.0f, 0.7f, 14, 6, true, 1.0,
                    getValues<T>({108.129, 109.37, 111.623, 111.468,
                                  12.9725, 11.6102, 16.4918, 16.9624,
                                  112.883, 113.124, 115.17, 118.213,
                                  4.49132, 4.30537, 8.75027, 8.8035,
                                  24.9778, 25.0318, 27.2283, 28.495,
                                  100.126, 101.409, 102.354, 106.125,
                                  108.129, 109.37, 111.623, 111.468,
                                  12.9725, 11.6102, 16.4918, 16.9624,
                                  112.883, 113.124, 115.17, 118.213,
                                  4.49132, 4.30537, 8.75027, 8.8035,
                                  24.9778, 25.0318, 27.2283, 28.495,
                                  100.126, 101.409, 102.354, 106.125}),
                    getValues<T>({0.922952, 0.90457, 0.886462, 0.826641, 0.698273, 0.691854,
                                  0.922952, 0.90457, 0.886462, 0.826641, 0.698273, 0.691854}),
                    {6, 6}
            },

            {
                    0.1f, 0.7f, 1000, 6, true, 1.0,
                    getValues<T>({108.129, 109.37, 111.623, 111.468,
                                  12.9725, 11.6102, 16.4918, 16.9624,
                                  112.883, 113.124, 115.17, 118.213,
                                  4.49132, 4.30537, 8.75027, 8.8035,
                                  24.9778, 25.0318, 27.2283, 28.495,
                                  100.126, 101.409, 102.354, 106.125,
                                  108.129, 109.37, 111.623, 111.468,
                                  12.9725, 11.6102, 16.4918, 16.9624,
                                  112.883, 113.124, 115.17, 118.213,
                                  4.49132, 4.30537, 8.75027, 8.8035,
                                  24.9778, 25.0318, 27.2283, 28.495,
                                  100.126, 101.409, 102.354, 106.125}),
                    getValues<T>({0.922952, 0.90457, 0.886462, 0.826641, 0.698273, 0.691854,
                                  0.922952, 0.90457, 0.886462, 0.826641, 0.698273, 0.691854}),
                    {6, 6}
            },
            {
                    1.0f, 0.7f, 14, 6, false, 1.0,
                    getValues<T>({13.4588, 10.9153, 17.7377, 17.9436,
                                  4.73698, 3.95806, 10.1254, 9.70525,
                                  89.5773, 90.0053, 92.9476, 95.3396,
                                  0, 1.02093, 6.00364, 6.21505,
                                  92.3608, 94.306, 96.3198, 98.4288,
                                  135.252, 137.7, 140.716, 143.09,
                                  13.4588, 10.9153, 17.7377, 17.9436,
                                  4.73698, 3.95806, 10.1254, 9.70525,
                                  89.5773, 90.0053, 92.9476, 95.3396,
                                  0, 1.02093, 6.00364, 6.21505,
                                  92.3608, 94.306, 96.3198, 98.4288,
                                  135.252, 137.7, 140.716, 143.09}),
                    getValues<T>({0.90457, 0.826641, 0.657244, 0.566372, 0.566189, 0.559075,
                                  0.90457, 0.826641, 0.657244, 0.566372, 0.566189, 0.559075}),
                    {6, 6}
            },
            {
                    0.0f, 0.7f, 1000, 6, false, 1.0,
                    getValues<T>({108.194, 109.556, 112.435, 111.701,
                                  13.4588, 10.9153, 17.7377, 17.9436,
                                  113.324, 113.186, 115.755, 119.82,
                                  4.73698, 3.95806, 10.1254, 9.70525,
                                  25.4666, 25.0477, 27.8424, 29.2425,
                                  100.188, 101.614, 102.532, 107.687,
                                  108.194, 109.556, 112.435, 111.701,
                                  13.4588, 10.9153, 17.7377, 17.9436,
                                  113.324, 113.186, 115.755, 119.82,
                                  4.73698, 3.95806, 10.1254, 9.70525,
                                  25.4666, 25.0477, 27.8424, 29.2425,
                                  100.188, 101.614, 102.532, 107.687}),
                    getValues<T>({0.922952, 0.90457, 0.886462, 0.826641, 0.698273, 0.691854,
                                  0.922952, 0.90457, 0.886462, 0.826641, 0.698273, 0.691854}),
                    {6, 6}
            }
    };
    return params;
}
};  // namespace

template <typename T, typename ROIS_NUM_T>
struct generate_proposals_test
        : public ::testing::TestWithParam<GenerateProposalsParamsWithLayout<T> > {
public:
    void test() {
        GenerateProposalsParams<T> param;
        format::type data_layout;
        std::tie(param, data_layout) = this->GetParam();
        const bool need_reorder = data_layout != format::bfyx;

        const auto data_type = type_to_data_type<T>::value;
        const auto rois_num_type = type_to_data_type<ROIS_NUM_T>::value;

        auto& engine = get_test_engine();

        const primitive_id input_im_info_id = "InputImInfo";
        const auto input_im_info = engine.allocate_memory({data_type, format::bfyx, tensor{batch(num_batches), feature(3)}});
        set_values(input_im_info, getValues<T>(im_info));

        const primitive_id input_anchors_id = "InputAnchors";
        auto input_anchors = engine.allocate_memory(
                {data_type, format::bfyx, tensor{batch(height), feature(width), spatial(4, number_of_anchors)}});
        set_values(input_anchors, getValues<T>(anchors));

        const primitive_id input_deltas_id = "InputDeltas";
        auto input_deltas = engine.allocate_memory(
                {data_type, format::bfyx,
                 tensor{batch(num_batches), feature(number_of_anchors * 4), spatial(width, height)}});
        set_values(input_deltas, getValues<T>(deltas));

        const primitive_id input_scores_id = "InputScores";
        auto input_scores = engine.allocate_memory(
                {data_type, format::bfyx, tensor{batch(num_batches), feature(number_of_anchors), spatial(width, height)}});
        set_values(input_scores, getValues<T>(scores));

        const primitive_id output_roi_scores_id = "OutputRoiScores";
        const layout rois_scores_layout{data_type, data_layout, tensor{batch(num_batches * param.post_nms_count)}};
        auto output_roi_scores = engine.allocate_memory(rois_scores_layout);

        const primitive_id output_rois_num_id = "OutputRoisNum";
        const layout rois_num_layout{rois_num_type, data_layout, tensor{batch(num_batches)}};
        auto output_rois_num = engine.allocate_memory(rois_num_layout);

        const primitive_id reorder_im_info_id = input_im_info_id + "Reordered";
        const primitive_id reorder_anchors_id = input_anchors_id + "Reordered";
        const primitive_id reorder_deltas_id = input_deltas_id + "Reordered";
        const primitive_id reorder_scores_id = input_scores_id + "Reordered";

        topology topology;

        topology.add(input_layout{input_im_info_id, input_im_info->get_layout()});
        topology.add(input_layout{input_anchors_id, input_anchors->get_layout()});
        topology.add(input_layout{input_deltas_id, input_deltas->get_layout()});
        topology.add(input_layout{input_scores_id, input_scores->get_layout()});
        topology.add(mutable_data{output_roi_scores_id, output_roi_scores});
        topology.add(mutable_data{output_rois_num_id, output_rois_num});

        topology.add(reorder(reorder_im_info_id, input_im_info_id, data_layout, data_type));
        topology.add(reorder(reorder_anchors_id, input_anchors_id, data_layout, data_type));
        topology.add(reorder(reorder_deltas_id, input_deltas_id, data_layout, data_type));
        topology.add(reorder(reorder_scores_id, input_scores_id, data_layout, data_type));

        const primitive_id generate_proposals_id = "generate_proposals";
        const std::vector<primitive_id> inputs{ reorder_im_info_id, reorder_anchors_id, reorder_deltas_id,
                                                reorder_scores_id, output_roi_scores_id, output_rois_num_id};
        const auto generate_proposals_primitive = generate_proposals{
            generate_proposals_id,
            inputs,
            param.min_size,
            param.nms_threshold,
            param.pre_nms_count,
            param.post_nms_count,
            param.normalized,
            param.nms_eta,
            rois_num_type};

        topology.add(generate_proposals_primitive);
        const primitive_id reorder_result_id = generate_proposals_id + "Reordered";
        topology.add(reorder(reorder_result_id, generate_proposals_id, format::bfyx, data_type));

        network network{engine, topology};

        network.set_input_data(input_im_info_id, input_im_info);
        network.set_input_data(input_anchors_id, input_anchors);
        network.set_input_data(input_deltas_id, input_deltas);
        network.set_input_data(input_scores_id, input_scores);

        const auto outputs = network.execute();

        const auto rois = outputs.at(reorder_result_id).get_memory();

        const cldnn::mem_lock<T> rois_ptr(rois, get_test_stream());
        ASSERT_EQ(rois_ptr.size(), num_batches * param.post_nms_count * 4);

        const auto get_plane_data = [&](const memory::ptr& mem, const data_types data_type, const layout& from_layout) {
            if (!need_reorder) {
                return mem;
            }
            cldnn::topology reorder_topology;
            reorder_topology.add(input_layout("data", from_layout));
            reorder_topology.add(reorder("plane_data", "data", format::bfyx, data_type));
            cldnn::network reorder_net{engine, reorder_topology};
            reorder_net.set_input_data("data", mem);
            const auto second_output_result = reorder_net.execute();
            const auto plane_data_mem = second_output_result.at("plane_data").get_memory();
            return plane_data_mem;
        };

        const cldnn::mem_lock<T> roi_scores_ptr(
                get_plane_data(output_roi_scores, data_type, rois_scores_layout), get_test_stream());
        ASSERT_EQ(roi_scores_ptr.size(), num_batches * param.post_nms_count);

        const cldnn::mem_lock<ROIS_NUM_T> rois_num_ptr(
                get_plane_data(output_rois_num, rois_num_type, rois_num_layout), get_test_stream());
        ASSERT_EQ(rois_num_ptr.size(), num_batches);

        const auto& expected_rois = param.expected_rois;
        const auto& expected_roi_scores = param.expected_roi_scores;
        const auto& expected_rois_num = param.expected_rois_num;

        for (size_t j = 0; j < expected_rois_num.size(); ++j) {
            EXPECT_EQ(expected_rois_num[j], rois_num_ptr[j]) << "j=" << j;
        }

        for (auto i = 0; i < param.post_nms_count; ++i) {
            EXPECT_NEAR(expected_roi_scores[i], roi_scores_ptr[i], getError<T>()) << "i=" << i;

            if (static_cast<float>(expected_roi_scores[i]) != 0.0f) {
                for (size_t coord = 0; coord < 4; ++coord) {
                    const auto roi_idx = i * 4 + coord;
                    EXPECT_NEAR(expected_rois[roi_idx], rois_ptr[roi_idx], getError<T>()) << "i=" << i << ", coord=" << coord;
                }
            }
        }
    }
};

using f32_i32 = generate_proposals_test<float, int32_t>;
TEST_P(f32_i32, f32_i32) {
    test();
}
INSTANTIATE_TEST_SUITE_P(
        generate_proposals_gpu_test,
        f32_i32,
        ::testing::Combine(
            ::testing::ValuesIn(getGenerateProposalsParams<float>()),
            ::testing::ValuesIn(layouts)
            ));

using f32_i64 = generate_proposals_test<float, int64_t>;
TEST_P(f32_i64, f32_i64) {
    test();
}
INSTANTIATE_TEST_SUITE_P(
        generate_proposals_gpu_test,
        f32_i64,
        ::testing::Combine(
                ::testing::ValuesIn(getGenerateProposalsParams<float>()),
                ::testing::ValuesIn(layouts)
        ));

using f16_i32 = generate_proposals_test<half_t, int32_t>;
TEST_P(f16_i32, f16_i32) {
    test();
}
INSTANTIATE_TEST_SUITE_P(
        generate_proposals_gpu_test,
        f16_i32,
        ::testing::Combine(
                ::testing::ValuesIn(getGenerateProposalsParams<half_t>()),
                ::testing::ValuesIn(layouts)
        ));

using f16_i64 = generate_proposals_test<half_t, int64_t>;
TEST_P(f16_i64, f16_i64) {
    test();
}
INSTANTIATE_TEST_SUITE_P(
        generate_proposals_gpu_test,
        f16_i64,
        ::testing::Combine(
                ::testing::ValuesIn(getGenerateProposalsParams<half_t>()),
                ::testing::ValuesIn(layouts)
        ));
