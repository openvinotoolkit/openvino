// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "ngraph/ngraph.hpp"

#include "shared_test_classes/single_layer/slice.hpp"

using namespace ngraph;

namespace LayerTestsDefinitions {

std::string Slice8LayerTest::getTestCaseName(const testing::TestParamInfo<Slice8Params> &obj) {
    std::vector<ov::test::InputShape> shapes;
    Slice8SpecificParams params;
    ov::element::Type_t netPrecision, inPrecision, outPrecision;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::map<std::string, std::string> additionalConfig;
    std::tie(params, netPrecision, inPrecision, outPrecision, inLayout, outLayout, targetName, additionalConfig) = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : params.shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : params.shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "start="   << ov::test::utils::vec2str(params.start) << "_";
    result << "stop="    << ov::test::utils::vec2str(params.stop) << "_";
    result << "step="    << ov::test::utils::vec2str(params.step) << "_";
    result << "axes="    << ov::test::utils::vec2str(params.axes) << "_";
    result << "netPRC="  << netPrecision << "_";
    result << "trgDev="  << targetName;
    return result.str();
}

void Slice8LayerTest::SetUp() {
    Slice8SpecificParams sliceParams;
    ov::test::ElementType netPrecision, inPrecision, outPrecision;
    InferenceEngine::Layout inLayout, outLayout;
    std::map<std::string, std::string> additionalConfig;
    std::tie(sliceParams, netPrecision, inPrecision, outPrecision, inLayout, outLayout, targetDevice, additionalConfig) = this->GetParam();

    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    init_input_shapes(sliceParams.shapes);
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));
    }
    auto sliceOp = ngraph::builder::makeSlice(params[0], sliceParams.start, sliceParams.stop, sliceParams.step, sliceParams.axes, netPrecision);

    ov::ResultVector results;
    for (int i = 0; i < sliceOp->get_output_size(); i++)
         results.push_back(std::make_shared<ov::op::v0::Result>(sliceOp->output(i)));
    function = std::make_shared<ngraph::Function>(results, params, "Slice-8");
}

}  // namespace LayerTestsDefinitions
