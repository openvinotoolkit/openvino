// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/experimental_detectron_generate_proposals_single_image.hpp>

using namespace cldnn;
using namespace ::tests;

namespace {
template <typename T>
struct ExperimentalDetectronGenerateProposalsSingleImageParams {
    float min_size;
    float nms_threshold;
    int64_t pre_nms_count;
    int64_t post_nms_count;
    std::vector<T> expected_rois;
    std::vector<T> expected_roi_scores;
};

template <typename T>
using ExperimentalDetectronGenerateProposalsSingleImageParamsWithLayout =
        std::tuple<ExperimentalDetectronGenerateProposalsSingleImageParams<T>, format::type, bool>;

constexpr size_t image_height = 150;
constexpr size_t image_width = 150;
constexpr float image_scale = 1.0f;
constexpr size_t height = 2;
constexpr size_t width = 6;
constexpr size_t number_of_channels = 3;

const std::vector<float> im_info{image_height, image_width, image_scale};

const std::vector<float> anchors{
        12.0, 68.0, 102.0, 123.0, 46.0, 80.0, 79.0, 128.0, 33.0, 71.0, 127.0, 86.0, 33.0, 56.0, 150.0, 73.0,
        5.0, 41.0, 93.0, 150.0, 74.0, 66.0, 106.0, 115.0, 17.0, 37.0, 87.0, 150.0, 31.0, 27.0, 150.0, 39.0,
        29.0, 23.0, 112.0, 123.0, 41.0, 37.0, 103.0, 150.0, 8.0, 46.0, 98.0, 111.0, 7.0, 69.0, 114.0, 150.0,
        70.0, 21.0, 150.0, 125.0, 54.0, 19.0, 132.0, 68.0, 62.0, 8.0, 150.0, 101.0, 57.0, 81.0, 150.0, 97.0,
        79.0, 29.0, 109.0, 130.0, 12.0, 63.0, 100.0, 150.0, 17.0, 33.0, 113.0, 150.0, 90.0, 78.0, 150.0, 111.0,
        47.0, 68.0, 150.0, 71.0, 66.0, 103.0, 111.0, 150.0, 4.0, 17.0, 112.0, 94.0, 12.0, 8.0, 119.0, 98.0,
        54.0, 56.0, 120.0, 150.0, 56.0, 29.0, 150.0, 31.0, 42.0, 3.0, 139.0, 92.0, 41.0, 65.0, 150.0, 130.0,
        49.0, 13.0, 143.0, 30.0, 40.0, 60.0, 150.0, 150.0, 23.0, 73.0, 24.0, 115.0, 56.0, 84.0, 107.0, 108.0,
        63.0, 8.0, 142.0, 125.0, 78.0, 37.0, 93.0, 144.0, 40.0, 34.0, 150.0, 46.0, 30.0, 21.0, 150.0, 120.0};

const std::vector<float> deltas{
        9.062256, 10.883133, 9.8441105, 12.694285, 0.41781136, 8.749107, 14.990341, 6.587644, 1.4206103,
        13.299262, 12.432549, 2.736371, 0.22732796, 6.3361835, 12.268727, 2.1009045, 4.771589, 2.5131326,
        5.610736, 9.3604145, 4.27379, 8.317948, 0.60510135, 6.7446275, 1.0207708, 1.1352817, 1.5785321,
        1.718335, 1.8093798, 0.99247587, 1.3233583, 1.7432803, 1.8534478, 1.2593061, 1.7394226, 1.7686696,
        1.647999, 1.7611449, 1.3119122, 0.03007332, 1.1106564, 0.55669737, 0.2546148, 1.9181818, 0.7134989,
        2.0407224, 1.7211134, 1.8565536, 14.562747, 2.8786168, 0.5927796, 0.2064463, 7.6794515, 8.672126,
        10.139171, 8.002429, 7.002932, 12.6314945, 10.550842, 0.15784842, 0.3194304, 10.752157, 3.709805,
        11.628928, 0.7136225, 14.619964, 15.177284, 2.2824087, 15.381494, 0.16618137, 7.507227, 11.173228,
        0.4923559, 1.8227729, 1.4749299, 1.7833921, 1.2363617, -0.23659119, 1.5737582, 1.779316, 1.9828427,
        1.0482665, 1.4900246, 1.3563544, 1.5341306, 0.7634312, 4.6216766e-05, 1.6161222, 1.7512476, 1.9363779,
        0.9195784, 1.4906164, -0.03244795, 0.681073, 0.6192401, 1.8033613, 14.146055, 3.4043705, 15.292292,
        3.5295358, 11.138999, 9.952057, 5.633434, 12.114562, 9.427372, 12.384038, 9.583308, 8.427233,
        15.293704, 3.288159, 11.64898, 9.350885, 2.0037227, 13.523184, 4.4176426, 6.1057625, 14.400079,
        8.248259, 11.815807, 15.713364, 1.0023532, 1.3203261, 1.7100681, 0.7407832, 1.09448, 1.7188418,
        1.4412547, 1.4862992, 0.74790007, 0.31571656, 0.6398838, 2.0236106, 1.1869069, 1.7265586, 1.2624544,
        0.09934269, 1.3508598, 0.85212964, -0.38968498, 1.7059708, 1.6533034, 1.7400402, 1.8123854, -0.43063712};

const std::vector<float> scores{
        0.7719922, 0.35906568, 0.29054508, 0.18124384, 0.5604661, 0.84750974, 0.98948747, 0.009793862, 0.7184191,
        0.5560748, 0.6952493, 0.6732593, 0.3306898, 0.6790913, 0.41128764, 0.34593266, 0.94296855, 0.7348507,
        0.24478768, 0.94024557, 0.05405676, 0.06466125, 0.36244348, 0.07942984, 0.10619422, 0.09412837, 0.9053611,
        0.22870538, 0.9237487, 0.20986171, 0.5067282, 0.29709867, 0.53138554, 0.189101, 0.4786443, 0.88421875};

const std::vector<format::type> layouts{
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
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
float getError<ov::float16>() {
    return 0.25;
}

template <typename T>
std::vector<ExperimentalDetectronGenerateProposalsSingleImageParams<T>> getExperimentalDetectronGenerateProposalsSingleImageParams() {
    std::vector<ExperimentalDetectronGenerateProposalsSingleImageParams<T>> params = {
            {
                    0.0f, 0.7f, 1000, 6,
                    getValues<T>({149.0, 149.0, 149.0, 149.0,
                                  149.0, 0.0, 149.0, 149.0,
                                  149.0, 60.8744, 149.0, 149.0,
                                  149.0, 61.8950, 149.0, 149.0,
                                  149.0, 149.0, 149.0, 149.0,
                                  149.0, 149.0, 149.0, 149.0}),
                    getValues<T>({0.989487, 0.942969, 0.940246, 0.923749, 0.905361, 0.884219})
            },
            {
                    1.5f, 0.4f, 1000, 10,
                    getValues<T>({43.171, 0.31823, 53.5592, 149,
                                  0, 75.2272, 149, 87.2278,
                                  141.058, 114.876, 149, 149,
                                  0, 146.297, 149, 149,
                                  0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0}),
                    getValues<T>({0.695249, 0.411288, 0.0941284, 0.0794298, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0})
            },
            {
                    5.0f, 0.71f, 10, 15,
                    getValues<T>({43.171, 0.31823, 53.5592, 149,
                                  0, 75.2272, 149, 87.2278,
                                  141.058, 114.876, 149, 149,
                                  149, 149, 149, 149,
                                  30.2866, 149, 149, 149,
                                  149, 149, 149, 149,
                                  149, 126.679, 149, 149,
                                  149, 6.53844, 149, 149,
                                  149, 0, 149, 149,
                                  149, 149, 149, 149,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0}),
                    getValues<T>({0.695249, 0.411288, 0.0941284, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
            }
    };
    return params;
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<ExperimentalDetectronGenerateProposalsSingleImageParamsWithLayout<T> > &param) {
        std::stringstream buf;
        ExperimentalDetectronGenerateProposalsSingleImageParams<T> p;
        format::type layout;
        bool is_caching_test;
        std::tie(p, layout, is_caching_test) = param.param;

        buf << "min_size=" << p.min_size << "_";
        buf << "nms_threshold=" << p.nms_threshold << "_";
        buf << "pre_nms_count=" << p.pre_nms_count << "_";
        buf << "post_nms_count=" << p.post_nms_count << "_";
        buf << "layout=" << fmt_to_str(layout);
        buf << "is_caching_test=" << is_caching_test;
        return buf.str();
    }
};
};  // namespace

template <typename T>
struct experimental_detectron_generate_proposals_single_image_test
        : public ::testing::TestWithParam<ExperimentalDetectronGenerateProposalsSingleImageParamsWithLayout<T> > {
public:
    void test() {
        ExperimentalDetectronGenerateProposalsSingleImageParams<T> param;
        format::type data_layout;
        bool is_caching_test;
        std::tie(param, data_layout, is_caching_test) = this->GetParam();
        const auto data_type = ov::element::from<T>();

        auto &engine = get_test_engine();

        const primitive_id input_im_info_id = "InputImInfo";
        const auto input_im_info = engine.allocate_memory({data_type, format::bfyx, tensor{batch(3)}});
        set_values(input_im_info, getValues<T>(im_info));

        const primitive_id input_anchors_id = "InputAnchors";
        auto input_anchors = engine.allocate_memory(
                {data_type, format::bfyx, tensor{batch(height * width * number_of_channels), feature(4)}});
        set_values(input_anchors, getValues<T>(anchors));

        const primitive_id input_deltas_id = "InputDeltas";
        auto input_deltas = engine.allocate_memory(
                {data_type, format::bfyx,
                 tensor{batch(number_of_channels * 4), feature(height), spatial(1, width)}});
        set_values(input_deltas, getValues<T>(deltas));

        const primitive_id input_scores_id = "InputScores";
        auto input_scores = engine.allocate_memory(
                {data_type, format::bfyx, tensor{batch(number_of_channels), feature(height), spatial(1, width)}});
        set_values(input_scores, getValues<T>(scores));

        const primitive_id output_roi_scores_id = "OutputRoiScores";
        const layout rois_scores_layout{data_type, data_layout, tensor{batch(param.post_nms_count)}};
        auto output_roi_scores = engine.allocate_memory(rois_scores_layout);

        const primitive_id reorder_im_info_id = input_im_info_id + "Reordered";
        const primitive_id reorder_anchors_id = input_anchors_id + "Reordered";
        const primitive_id reorder_deltas_id = input_deltas_id + "Reordered";
        const primitive_id reorder_scores_id = input_scores_id + "Reordered";

        topology topology;

        topology.add(input_layout(input_im_info_id, input_im_info->get_layout()));
        topology.add(input_layout(input_anchors_id, input_anchors->get_layout()));
        topology.add(input_layout(input_deltas_id, input_deltas->get_layout()));
        topology.add(input_layout(input_scores_id, input_scores->get_layout()));
        topology.add(mutable_data(output_roi_scores_id, output_roi_scores));

        topology.add(reorder(reorder_im_info_id, input_info(input_im_info_id), data_layout, data_type));
        topology.add(reorder(reorder_anchors_id, input_info(input_anchors_id), data_layout, data_type));
        topology.add(reorder(reorder_deltas_id, input_info(input_deltas_id), data_layout, data_type));
        topology.add(reorder(reorder_scores_id, input_info(input_scores_id), data_layout, data_type));

        const primitive_id edgpsi_id = "experimental_detectron_generate_proposals_single_image";
        const auto edgpsi_primitive = experimental_detectron_generate_proposals_single_image{edgpsi_id,
                                                                                             input_info(reorder_im_info_id),
                                                                                             input_info(reorder_anchors_id),
                                                                                             input_info(reorder_deltas_id),
                                                                                             input_info(reorder_scores_id),
                                                                                             input_info(output_roi_scores_id),
                                                                                             param.min_size,
                                                                                             param.nms_threshold,
                                                                                             param.pre_nms_count,
                                                                                             param.post_nms_count};
        topology.add(edgpsi_primitive);

        const primitive_id reorder_result_id = edgpsi_id + "Reordered";
        topology.add(reorder(reorder_result_id, input_info(edgpsi_primitive), format::bfyx, data_type));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data(input_im_info_id, input_im_info);
        network->set_input_data(input_anchors_id, input_anchors);
        network->set_input_data(input_deltas_id, input_deltas);
        network->set_input_data(input_scores_id, input_scores);

        const auto outputs = network->execute();

        const auto rois = outputs.at(reorder_result_id).get_memory();

        const cldnn::mem_lock<T> rois_ptr(rois, get_test_stream());
        ASSERT_EQ(rois_ptr.size(), param.post_nms_count * 4);

        cldnn::topology reorder_topology;
        reorder_topology.add(input_layout("scores", rois_scores_layout));
        reorder_topology.add(reorder("plane_scores", input_info("scores"), format::bfyx, data_type));
        cldnn::network reorder_net{engine, reorder_topology, get_test_default_config(engine)};
        reorder_net.set_input_data("scores", output_roi_scores);
        const auto second_output_result = reorder_net.execute();
        const auto plane_data_mem = second_output_result.at("plane_scores").get_memory();

        const cldnn::mem_lock<T> roi_scores_ptr(plane_data_mem, get_test_stream());
        ASSERT_EQ(roi_scores_ptr.size(), param.post_nms_count);

        const auto &expected_roi_scores = param.expected_roi_scores;
        const auto &expected_rois = param.expected_rois;
        for (int64_t i = 0; i < param.post_nms_count; ++i) {
            if (!is_caching_test) {
                ASSERT_NEAR(expected_roi_scores[i], roi_scores_ptr[i], 0.001) << "i=" << i;
            }

            // order of proposals with zero scores is not guaranteed (to be precise,
            // it is not guaranteed for any equal score values)
            if (static_cast<float>(expected_roi_scores[i]) != 0.0f) {
                for (size_t coord = 0; coord < 4; ++coord) {
                    const auto roi_idx = i * 4 + coord;
                    ASSERT_NEAR(expected_rois[roi_idx], rois_ptr[roi_idx], getError<T>()) << "i=" << i << ", coord=" << coord;
                }
            }
        }
    }
};

using experimental_detectron_generate_proposals_single_image_test_f32 = experimental_detectron_generate_proposals_single_image_test<float>;
TEST_P(experimental_detectron_generate_proposals_single_image_test_f32, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}
using experimental_detectron_generate_proposals_single_image_test_f16 = experimental_detectron_generate_proposals_single_image_test<ov::float16>;
TEST_P(experimental_detectron_generate_proposals_single_image_test_f16, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}

INSTANTIATE_TEST_SUITE_P(
        experimental_detectron_generate_proposals_single_image_gpu_test,
        experimental_detectron_generate_proposals_single_image_test_f32,
        ::testing::Combine(
                ::testing::ValuesIn(getExperimentalDetectronGenerateProposalsSingleImageParams<float>()),
                ::testing::Values(format::bfyx),
                ::testing::Values(false)
        ),
        PrintToStringParamName()
);

INSTANTIATE_TEST_SUITE_P(
        experimental_detectron_generate_proposals_single_image_gpu_test,
        experimental_detectron_generate_proposals_single_image_test_f16,
        ::testing::Combine(
                ::testing::ValuesIn(getExperimentalDetectronGenerateProposalsSingleImageParams<ov::float16>()),
                ::testing::Values(format::bfyx),
                ::testing::Values(false)
        ),
        PrintToStringParamName()
);

INSTANTIATE_TEST_SUITE_P(
        export_import,
        experimental_detectron_generate_proposals_single_image_test_f16,
        ::testing::Combine(
                ::testing::Values(getExperimentalDetectronGenerateProposalsSingleImageParams<ov::float16>()[0]),
                ::testing::Values(format::bfyx),
                ::testing::Values(true)
        ),
        PrintToStringParamName()
);

INSTANTIATE_TEST_SUITE_P(
        experimental_detectron_generate_proposals_single_image_gpu_test_all_formats,
        experimental_detectron_generate_proposals_single_image_test_f32,
        ::testing::Combine(
                ::testing::Values(
                    ExperimentalDetectronGenerateProposalsSingleImageParams<float>{
                        5.0f, 0.71f, 10, 15,
                        {43.171, 0.31823, 53.5592, 149,
                         0, 75.2272, 149, 87.2278,
                         141.058, 114.876, 149, 149,
                         149, 149, 149, 149,
                         30.2866, 149, 149, 149,
                         149, 149, 149, 149,
                         149, 126.679, 149, 149,
                         149, 6.53844, 149, 149,
                         149, 0, 149, 149,
                         149, 149, 149, 149,
                         0, 0, 0, 0,
                         0, 0, 0, 0,
                         0, 0, 0, 0,
                         0, 0, 0, 0,
                         0, 0, 0, 0},
                        {0.695249, 0.411288, 0.0941284, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}
                    ),
                ::testing::ValuesIn(layouts),
                ::testing::Values(false)
        ),
        PrintToStringParamName()
);
