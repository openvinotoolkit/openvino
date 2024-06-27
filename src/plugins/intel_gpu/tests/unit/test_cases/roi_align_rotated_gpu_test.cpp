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

constexpr float EPS = 2e-3f;

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
    int32_t pooledH;
    int32_t pooledW;
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
                                    int32_t pooledH,
                                    int32_t pooledW,
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
    ret.rois = AllocateTensor<T>({numOfRois, rois_second_dim_size}, cldnn::format::bfyx, roisVals);
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
            ASSERT_TRUE(are_equal(wanted_output_ptr[i], output_ptr[i], EPS));
    }

private:
    engine& engine_ = get_test_engine();
};

template <ov::element::Type_t ET>
std::vector<ROIAlignRotatedParams> generateParams() {
    using T = typename ov::element_type_traits<ET>::value_type;
    std::vector<ROIAlignRotatedParams> params;
    // NOTE: expected output were generated using mmvc roi_align_rotated implementation.
#define TEST_DATA(input_shape,                            \
                  pooled_height,                          \
                  pooled_width,                           \
                  spatial_scale,                          \
                  sampling_ratio,                         \
                  clockwise,                              \
                  input_data,                             \
                  rois_data,                              \
                  batch_indices_data,                     \
                  expected_output,                        \
                  description)                            \
    params.push_back(PrepareParams<T>(input_shape,        \
                                      pooled_height,      \
                                      pooled_width,       \
                                      spatial_scale,      \
                                      sampling_ratio,     \
                                      clockwise,          \
                                      input_data,         \
                                      rois_data,          \
                                      batch_indices_data, \
                                      expected_output,    \
                                      description));

#include "unit_test_utils/tests_data/roi_align_rotated_data.h"
#undef TEST_DATA
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
