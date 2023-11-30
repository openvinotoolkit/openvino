// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multi_crops_to_concat.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string MultiCropsToConcatTest::getTestCaseName(const testing::TestParamInfo<MultiCropsToConcatParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::vector<size_t> inputShape;
    std::vector<std::pair<int64_t, int64_t>> offsets;
    std::tie(netPrecision, targetDevice, inputShape, offsets, configuration) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    result << "offset=";
    for (auto offset : offsets) {
        result << "(" << offset.first << "," << offset.second << ")";
    }
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void MultiCropsToConcatTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    std::vector<size_t> inputShape;
    std::vector<std::pair<int64_t, int64_t>> offsets;
    std::tie(netPrecision, targetDevice, inputShape, offsets, tempConfig) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    ov::Shape const_shape_crop1 = {2};
    auto begin_node_crop1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape_crop1, std::vector<int64_t>{ 0, offsets[0].first });
    auto end_node_crop1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape_crop1, std::vector<int64_t>{ 1, offsets[0].second });
    auto strideN_node_crop1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape_crop1, std::vector<int64_t>{ 1, 1 });
    auto crop1 = std::make_shared<ov::op::v1::StridedSlice>(params[0],
                                                            begin_node_crop1,
                                                            end_node_crop1,
                                                            strideN_node_crop1,
                                                            std::vector<int64_t>{ 1, 0 },
                                                            std::vector<int64_t>{ 1, 0 },
                                                            std::vector<int64_t>{ 0, 0 },
                                                            std::vector<int64_t>{ 0, 0 },
                                                            std::vector<int64_t>{ 0, 0 });

    ov::Shape const_shape_crop2 = {2};
    auto begin_node_crop2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape_crop2, std::vector<int64_t>{ 0, offsets[1].first });
    auto end_node_crop2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape_crop2, std::vector<int64_t>{ 1, offsets[1].second });
    auto strideN_node_crop2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape_crop2, std::vector<int64_t>{ 1, 1 });
    auto crop2 = std::make_shared<ov::op::v1::StridedSlice>(params[0],
                                                            begin_node_crop2,
                                                            end_node_crop2,
                                                            strideN_node_crop2,
                                                            std::vector<int64_t>{ 1, 0 },
                                                            std::vector<int64_t>{ 1, 0 },
                                                            std::vector<int64_t>{ 0, 0 },
                                                            std::vector<int64_t>{ 0, 0 },
                                                            std::vector<int64_t>{ 0, 0 });

    auto concat1 = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{crop1, crop2}, 1);
    std::shared_ptr<ov::op::v0::Result> result;

    // Case with 3 crops
    if (offsets.size() == 3) {
        ov::Shape const_shape_crop3 = {2};
        auto begin_node_crop3 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape_crop3, std::vector<int64_t>{ 0, offsets[2].first });
        auto end_node_crop3 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape_crop3, std::vector<int64_t>{ 1, offsets[2].second });
        auto strideN_node_crop3 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape_crop3, std::vector<int64_t>{ 1, 1 });
        auto crop3 = std::make_shared<ov::op::v1::StridedSlice>(params[0],
                                                                begin_node_crop3,
                                                                end_node_crop3,
                                                                strideN_node_crop3,
                                                                std::vector<int64_t>{ 1, 0 },
                                                                std::vector<int64_t>{ 1, 0 },
                                                                std::vector<int64_t>{ 0, 0 },
                                                                std::vector<int64_t>{ 0, 0 },
                                                                std::vector<int64_t>{ 0, 0 });

        auto concat2 = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{crop1, crop2}, 1);
        result = std::make_shared<ngraph::opset8::Result>(concat2);
    } else {
        result = std::make_shared<ngraph::opset8::Result>(concat1);
    }
    function = std::make_shared<ngraph::Function>(result, params, "InputSplitConcatTest");
}
}  // namespace SubgraphTestsDefinitions
