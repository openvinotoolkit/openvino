// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph_functions/builders.hpp>
#include "shared_test_classes/subgraph/mvn_fq_mvn.hpp"

namespace SubgraphTestsDefinitions {

    std::string MvnFqMvnSubgraphTest::getTestCaseName(testing::TestParamInfo<fqSubgraphTestParamsSet> obj) {
        fqSpecificParams fqParams;
        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision dataPrecision, axesPrecision;
        std::vector<int> axes;
        bool normalizeVariance;
        float eps;
        std::string epsMode;
        std::string targetDevice;
        std::tie(fqParams, inputShapes, dataPrecision, axesPrecision, axes, normalizeVariance, eps, epsMode, targetDevice) = obj.param;

        size_t levels;
        std::vector<size_t> constShape;
        std::vector<float> inputArg;
        std::tie(levels, constShape, inputArg) = fqParams;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "DataPrc=" << dataPrecision.name() << "_";
        result << "AxPrc=" << axesPrecision.name() << "_";
        result << "Ax=" << CommonTestUtils::vec2str(axes) << "_";
        result << "NormVariance=" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
        result << "Eps=" << eps << "_";
        result << "EM=" << epsMode << "_";
        result << "LEVELS=" << levels << "_";
        result << "CS=" << CommonTestUtils::vec2str(constShape) << "_";
        if (inputArg.size() == 3) {
            result << "_inputArg=" << inputArg[0] << "_" << inputArg[1] << "_" << inputArg[2];
        }
        result << "TargetDevice=" << targetDevice;
        return result.str();
    }

    void MvnFqMvnSubgraphTest::SetUp() {
        fqSpecificParams fqParams;
        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision dataPrecision, axesPrecision;
        std::vector<int> axes;
        bool normalizeVariance;
        float eps;
        std::string epsMode;
        std::tie(fqParams, inputShapes, dataPrecision, axesPrecision, axes, normalizeVariance, eps, epsMode, targetDevice) = this->GetParam();

        size_t levels;
        std::vector<size_t> constShape;
        std::vector<float> inputArg;
        std::tie(levels, constShape, inputArg) = fqParams;
        if (inputArg.size() == 3) {
            inputDataMin = inputArg[0];
            inputDataMax = inputArg[1];
            inputDataResolution = inputArg[2];
        }

        auto dataType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dataPrecision);
        auto axesType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(axesPrecision);

        auto params = ngraph::builder::makeParams(dataType, {inputShapes});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto axesNode = ngraph::builder::makeConstant(axesType, ngraph::Shape{axes.size()}, axes);
        auto mvn1 = ngraph::builder::makeMVN6(paramOuts[0], axesNode, normalizeVariance, eps, epsMode);

        auto FQNode = ngraph::builder::makeFakeQuantize(mvn1, ngraph::element::f32, levels, constShape,
                                                        { inputDataMin }, { inputDataMax }, { inputDataMin }, { inputDataMax });

        auto mvn2 = ngraph::builder::makeMVN6(FQNode, axesNode, normalizeVariance, eps, epsMode);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mvn2)};
        function = std::make_shared<ngraph::Function>(results, params, "MvnFqMvnSubgraph");
    }

InferenceEngine::Blob::Ptr MvnFqMvnSubgraphTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution,
                                            seed);
}
} // namespace SubgraphTestsDefinitions
