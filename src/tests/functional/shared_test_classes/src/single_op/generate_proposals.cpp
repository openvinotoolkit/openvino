// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/generate_proposals.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/data_utils.hpp"

namespace ov {
namespace test {
std::string GenerateProposalsLayerTest::getTestCaseName(
        const testing::TestParamInfo<GenerateProposalsTestParams>& obj) {
    std::vector<InputShape> shapes;
    ov::op::v9::GenerateProposals::Attributes attributes;
    ov::element::Type model_type;
    ov::element::Type roi_num_type;
    std::string targetName;
    std::tie(
        shapes,
        attributes.min_size,
        attributes.nms_threshold,
        attributes.post_nms_count,
        attributes.pre_nms_count,
        attributes.normalized,
        model_type,
        roi_num_type,
        targetName) = obj.param;

    std::ostringstream result;
    using ov::test::operator<<;
    result << "im_info=" << shapes[0] << "_";
    result << "anchors=" << shapes[1] << "_";
    result << "deltas=" << shapes[2] << "_";
    result << "scores=" << shapes[3] << "_";

    using ov::test::operator<<;
    result << "attributes={";
    result << "score_threshold=" << attributes.min_size << "_";
    result << "nms_threshold=" << attributes.nms_threshold << "_";
    result << "post_nms_count=" << attributes.post_nms_count << "_";
    result << "pre_nms_count=" << attributes.pre_nms_count;
    result << "normalized=" << attributes.normalized;
    result << "nms_eta=" << attributes.nms_eta;
    result << "}_";

    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "roiNumPRC=" << roi_num_type.get_type_name() << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void GenerateProposalsLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::op::v9::GenerateProposals::Attributes attributes;
    ov::element::Type model_type;
    ov::element::Type roi_num_type;
    std::tie(
        shapes,
        attributes.min_size,
        attributes.nms_threshold,
        attributes.post_nms_count,
        attributes.pre_nms_count,
        attributes.normalized,
        model_type,
        roi_num_type,
        targetDevice) = this->GetParam();

    inType = outType = model_type;
    if (targetDevice == ov::test::utils::DEVICE_GPU) {
        if (model_type == element::Type_t::f16) {
            abs_threshold = 0.2;
        } else {
            abs_threshold = 0.00009;
        }
    }

    init_input_shapes(shapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }

    auto generate_proposals = std::make_shared<ov::op::v9::GenerateProposals>(
        params[0], // im_info
        params[1], // anchors
        params[2], // deltas
        params[3], // scores
        attributes,
        roi_num_type);
    function = std::make_shared<ov::Model>(
        generate_proposals->outputs(),
        params,
        "GenerateProposals");
}

void GenerateProposalsLayerTest::compare(const std::vector<ov::Tensor>& expected,
                                         const std::vector<ov::Tensor>& actual) {
    if (targetDevice != ov::test::utils::DEVICE_GPU) {
        SubgraphBaseTest::compare(expected, actual);
        return;
    }

    const auto outputsNum = expected.size();
    ASSERT_EQ(outputsNum, 3);
    ASSERT_EQ(outputsNum, actual.size());
    ASSERT_EQ(outputsNum, function->get_results().size());

    // actual outputs 0 (rois) and 1 (roi_scores) may be padded with zeros
    for (size_t i = 0; i < 2; ++i) {
        const auto expectedNumRois = expected[i].get_shape()[0];
        const auto actualNumRois = actual[i].get_shape()[0];
        ASSERT_LE(expectedNumRois, actualNumRois);

        const auto actualBuffer = static_cast<uint8_t*>(actual[i].data());
        const auto expectedBuffer = static_cast<uint8_t*>(expected[i].data());
        const auto outputSize = i == 0 ? 4 : 1;

        rel_threshold = ov::test::utils::tensor_comparation::calculate_default_rel_threshold(
            expected[i].get_element_type(), actual[i].get_element_type());

        if (outType == ov::element::f32) {
            ov::test::utils::compare_raw_data(reinterpret_cast<const float*>(expectedBuffer),
                                              reinterpret_cast<const float*>(actualBuffer),
                                              expectedNumRois * outputSize,
                                              rel_threshold,
                                              abs_threshold);
        } else {
            ov::test::utils::compare_raw_data(reinterpret_cast<const float16*>(expectedBuffer),
                                              reinterpret_cast<const float16*>(actualBuffer),
                                              expectedNumRois * outputSize,
                                              rel_threshold,
                                              abs_threshold);
        }

        if (expectedNumRois < actualNumRois) {
            if (outType == ov::element::f32) {
                const auto fBuffer = static_cast<const float*>(actual[i].data());
                for (size_t j = expectedNumRois * outputSize; j < actualNumRois * outputSize; ++j) {
                    ASSERT_TRUE(fBuffer[j] == 0.0f)
                        << "Expected 0.0, actual: " << fBuffer[j] << " at index: " << j << ", output: " << i;
                }
            } else {
                const float16 zero{0};
                const auto fBuffer = static_cast<const float16*>(actual[i].data());
                for (size_t j = expectedNumRois * outputSize; j < actualNumRois * outputSize; ++j) {
                    ASSERT_TRUE(fBuffer[j] == zero)
                        << "Expected 0.0, actual: " << fBuffer[j] << " at index: " << j << ", output: " << i;
                }
            }
        }
    }

    // output 2 - rois_num
    ov::test::utils::compare(expected[2], actual[2], abs_threshold, rel_threshold);
}
} // namespace test
} // namespace ov
