// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/experimental_detectron_topkrois.hpp"

namespace ov {
namespace test {
std::string ExperimentalDetectronTopKROIsLayerTest::getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronTopKROIsTestParams>& obj) {
    std::vector<InputShape> shapes;
    int64_t max_rois;
    ElementType model_type;
    std::string device_name;
    std::tie(shapes, max_rois, model_type, device_name) = obj.param;

    std::ostringstream result;
    if (shapes.front().first.size() != 0) {
        result << "IS=(";
        for (const auto &shape : shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result.seekp(-1, result.cur);
        result << ")_";
    }
    result << "TS=";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "max_rois=" << max_rois << "_";
    result << "netPRC=" << model_type << "_";
    result << "trgDev=" << device_name;
    return result.str();
}

void ExperimentalDetectronTopKROIsLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    int64_t max_rois;
    ElementType model_type;
    std::string targetName;
    std::tie(shapes, max_rois, model_type, targetName) = this->GetParam();

    inType = outType = model_type;
    targetDevice = targetName;

    init_input_shapes(shapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }

    auto experimentalDetectronTopKROIs = std::make_shared<ov::op::v6::ExperimentalDetectronTopKROIs>(params[0], params[1], max_rois);
    function = std::make_shared<ov::Model>(ov::OutputVector {experimentalDetectronTopKROIs->output(0)}, params, "ExperimentalDetectronTopKROIs");
}
} // namespace test
} // namespace ov
