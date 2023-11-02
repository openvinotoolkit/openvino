// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/generate_proposals.hpp"

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
        "GenerateProposals");
}
} // namespace test
} // namespace ov
