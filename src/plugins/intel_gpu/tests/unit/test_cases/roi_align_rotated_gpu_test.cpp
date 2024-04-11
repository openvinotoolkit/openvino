// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/roi_align.hpp>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

constexpr double PI = 3.141592653589793238462643383279;

// Allocates tensoer with given shape and data.
template <typename TDataType>
memory::ptr AllocateTensor(ov::PartialShape shape, cldnn::format fmt, const std::vector<TDataType>& data) {
    const layout lo = {shape, ov::element::from<TDataType>(), fmt};
    EXPECT_EQ(lo.get_linear_size(), data.size());
    memory::ptr tensor = get_test_engine().allocate_memory(lo);
    set_values<TDataType>(tensor, data);
    return tensor;
}

struct ROIAlignRotatedParams {
    size_t pooledH;
    size_t pooledW;
    float spatialScale;
    int32_t samplingRatio;
    bool clockwise;
    std::string testcaseName;
    memory::ptr input;
    memory::ptr rois;
    memory::ptr roiBatchIdxs;
    memory::ptr expectedOutput;
};

template <typename T>
ROIAlignRotatedParams PrepareParams(const ov::PartialShape& inputShape,
                                    size_t pooledH,
                                    size_t pooledW,
                                    float spatialScale,
                                    int32_t samplingRatio,
                                    bool clockwise,
                                    const std::vector<T>& inputValues,
                                    const std::vector<T>& roisVals,
                                    const std::vector<int32_t>& roiBatchIdx,
                                    const std::vector<float>& expectedValues,
                                    const std::string& testcaseName) {
    ROIAlignRotatedParams ret;

    constexpr ov::Dimension::value_type rois_second_dim_size = 5;  //< By definition of the ROIAlignRotated op

    const ov::Dimension::value_type numOfRois = roisVals.size() / rois_second_dim_size;
    const ov::Dimension::value_type channels = static_cast<ov::Dimension::value_type>(inputShape[1].get_length());

    ret.pooledH = pooledH;
    ret.pooledW = pooledW;
    ret.spatialScale = spatialScale;
    ret.samplingRatio = samplingRatio;
    ret.clockwise = clockwise;
    ret.testcaseName = testcaseName;

    ret.input = AllocateTensor<T>(inputShape, cldnn::format::bfyx, inputValues);
    ret.rois = AllocateTensor<T>({numOfRois, 5}, cldnn::format::bfyx, roisVals);
    ret.roiBatchIdxs = AllocateTensor<int32_t>({numOfRois}, cldnn::format::bfyx, roiBatchIdx);
    ret.expectedOutput = AllocateTensor<float>({numOfRois,
                                                channels,
                                                static_cast<ov::Dimension::value_type>(ret.pooledH),
                                                static_cast<ov::Dimension::value_type>(ret.pooledW)},
                                               cldnn::format::bfyx,
                                               expectedValues);

    return ret;
}

class roi_align_rotated_test : public ::testing::TestWithParam<ROIAlignRotatedParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ROIAlignRotatedParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "_type_" << param.input->get_layout().data_type;
        result << "_" << param.testcaseName;
        return result.str();
    }

    void Execute(const ROIAlignRotatedParams& params) {
        // Prepare the network.
        auto stream = get_test_stream_ptr(get_test_default_config(engine_));

        topology topology;
        topology.add(input_layout("input", params.input->get_layout()));
        topology.add(input_layout("rois", params.rois->get_layout()));
        topology.add(input_layout("roi_ind", params.roiBatchIdxs->get_layout()));
        topology.add(roi_align("roi_align",
                               {input_info("input"), input_info("rois"), input_info("roi_ind")},
                               params.pooledH,
                               params.pooledW,
                               params.samplingRatio,
                               params.spatialScale,
                               roi_align::PoolingMode::avg,
                               roi_align::AlignedMode::asymmetric,
                               roi_align::ROIMode::rotated,
                               params.clockwise));
        topology.add(reorder("out", input_info("roi_align"), cldnn::format::bfyx, data_types::f32));

        cldnn::network::ptr network = get_network(engine_, topology, get_test_default_config(engine_), stream, false);

        network->set_input_data("input", params.input);
        network->set_input_data("rois", params.rois);
        network->set_input_data("roi_ind", params.roiBatchIdxs);

        // Run and check results.
        auto outputs = network->execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());
        cldnn::mem_lock<float> wanted_output_ptr(params.expectedOutput, get_test_stream());

        ASSERT_EQ(output->get_layout(), params.expectedOutput->get_layout());
        ASSERT_EQ(output_ptr.size(), wanted_output_ptr.size());
        for (size_t i = 0; i < output_ptr.size(); ++i)
            ASSERT_TRUE(are_equal(wanted_output_ptr[i], output_ptr[i], 2e-3));
    }

private:
    engine& engine_ = get_test_engine();
};

template <ov::element::Type_t ET>
std::vector<ROIAlignRotatedParams> generateParams() {
    using T = typename ov::element_type_traits<ET>::value_type;
    std::vector<ROIAlignRotatedParams> params;
    // NOTE: expected output were generated using mmvc roi_align_rotated implementation.
    params.push_back(PrepareParams<T>(
        {2, 1, 8, 8},
        2,
        2,
        1.0f,
        2,
        true,
        {0,  1, 8, 5, 5,  2, 0,  7, 7, 10, 4, 5, 9,  0, 0,  5, 7, 0, 4, 0, 4, 7, 6, 10, 9,  5, 1,  7, 4, 7, 10, 8,
         2,  0, 8, 3, 6,  8, 10, 4, 2, 10, 7, 8, 7,  0, 6,  9, 2, 4, 8, 5, 2, 3, 3, 1,  5,  9, 10, 0, 9, 5, 5,  3,
         10, 5, 2, 0, 10, 0, 5,  4, 3, 10, 5, 5, 10, 0, 8,  8, 9, 1, 0, 7, 9, 6, 8, 7,  10, 9, 2,  3, 3, 5, 6,  9,
         4,  9, 2, 4, 5,  5, 3,  1, 1, 6,  8, 0, 5,  5, 10, 8, 6, 9, 6, 9, 1, 2, 7, 1,  1,  3, 0,  4, 0, 7, 10, 2},
        {3.5, 3.5, 2, 2, 0, 3.5, 3.5, 2, 2, 0},
        {0, 1},
        {3, 3.75, 4.75, 5, 3, 5.5, 2.75, 3.75},
        "roi_align_rotated_angle_0"));
    params.push_back(PrepareParams<T>({1, 1, 2, 2},
                                      2,
                                      2,
                                      1.0f,
                                      2,
                                      true,
                                      {1, 2, 3, 4},
                                      {0.5, 0.5, 1, 1, 0},
                                      {0},
                                      {1.0, 1.25, 1.5, 1.75},
                                      "roi_align_rotated_simple_angle_0"));
    params.push_back(PrepareParams<T>({1, 1, 2, 2},
                                      2,
                                      2,
                                      1.0f,
                                      2,
                                      false,
                                      {1, 2, 3, 4},
                                      {0.5, 0.5, 1, 1, PI / 2},
                                      {0},
                                      {1.5, 1.0, 1.75, 1.25},
                                      "roi_align_rotated_simple_angle_PI/2"));
    params.push_back(PrepareParams<T>({1, 1, 2, 2},
                                      2,
                                      2,
                                      1.0f,
                                      2,
                                      false,
                                      {1, 2, 3, 4},
                                      {0.5, 0.5, 1, 1, PI, 0.5, 0.5, 1, 1, 2 * PI},
                                      {0, 0},
                                      {1.75, 1.5, 1.25, 1.0, 1.0, 1.25, 1.5, 1.75},
                                      "roi_align_rotated_batch_idx_test"));
    params.push_back(PrepareParams<T>({1, 2, 2, 2},
                                      2,
                                      2,
                                      1.0f,
                                      2,
                                      false,
                                      {1, 2, 3, 4, 4, 3, 2, 1},
                                      {0.5, 0.5, 1, 1, 0},
                                      {0},
                                      {1.0, 1.25, 1.5, 1.75, 4.0, 3.75, 3.5, 3.25},
                                      "roi_align_rotated_channels_test"));
    params.push_back(
        PrepareParams<T>({1, 1, 5, 5},
                         3,
                         1,
                         1.0f,
                         2,
                         true,
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                         {1, 1, 4, 4, 0},
                         {0},
                         {0.8750, 4.2500, 10.9167},
                         "roi_align_rotated_box_outside_feature_map_top_left"));
    params.push_back(
        PrepareParams<T>({1, 1, 5, 5},
                         3,
                         1,
                         1.0f,
                         2,
                         true,
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                         {1, 1, 4, 4, PI / 4},
                         {0},
                         {2.6107, 4.6642, 6.8819},
                         "roi_align_rotated_box_outside_feature_map_top_left_angle_PI/4"));
    params.push_back(
        PrepareParams<T>({1, 1, 5, 5},
                         3,
                         1,
                         1.0f,
                         2,
                         true,
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                         {5, 5, 4, 4, 0},
                         {0},
                         {10.1667, 12.2500, 0.0},
                         "roi_align_rotated_box_outside_feature_map_bottom_right_angle_0"));
    params.push_back(
        PrepareParams<T>({1, 1, 5, 5},
                         3,
                         1,
                         1.0f,
                         2,
                         true,
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                         {5, 5, 1, 5, PI / 4},
                         {0},
                         {0.0, 25.0, 0.0},
                         "roi_align_rotated_box_outside_feature_map_bottom_right_angle_PI/4"));
    params.push_back(
        PrepareParams<T>({1, 1, 5, 5},
                         2,
                         2,
                         1.0f,
                         0,
                         true,
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                         {3, 3, 4, 4, 0},
                         {0},
                         {10.0, 12.0, 20.0, 22.0},
                         "roi_align_rotated_box_outside_sampling_ratio_auto"));
    params.push_back(
        PrepareParams<T>({1, 1, 5, 5},
                         2,
                         2,
                         0.25f,
                         0,
                         true,
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                         {3, 3, 4, 4, 0},
                         {0},
                         {1.0, 1.5, 3.5, 4.0},
                         "roi_align_rotated_box_outside_sampling_ratio_auto_scale_0.25"));
    params.push_back(
        PrepareParams<T>({1, 1, 5, 5},
                         2,
                         2,
                         2.0f,
                         0,
                         true,
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                         {3, 3, 4, 4, 0},
                         {0},
                         {20.5, 0.0, 0.0, 0.0},
                         "roi_align_rotated_box_outside_sampling_ratio_auto_scale_2"));
    params.push_back(
        PrepareParams<T>({1, 1, 5, 5},
                         5,
                         2,
                         0.78f,
                         0,
                         false,
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                         {3, 1, 4, 2, PI / 3},
                         {0},
                         {5.1271, 1.2473, 6.1773, 2.9598, 7.2275, 3.2300, 8.2777, 3.7458, 9.3279, 4.4060},
                         "roi_align_rotated_all_features"));
    return params;
}

std::vector<ROIAlignRotatedParams> generateCombinedParams() {
    const std::vector<std::vector<ROIAlignRotatedParams>> generatedParams{generateParams<ov::element::Type_t::f16>(),
                                                                          generateParams<ov::element::Type_t::f32>()};

    std::vector<ROIAlignRotatedParams> combinedParams;
    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

}  // namespace

TEST_P(roi_align_rotated_test, ref_comp) {
    Execute(GetParam());
}

INSTANTIATE_TEST_SUITE_P(roi_align_rotated_test_suit,
                         roi_align_rotated_test,
                         testing::ValuesIn(generateCombinedParams()),
                         roi_align_rotated_test::getTestCaseName);
