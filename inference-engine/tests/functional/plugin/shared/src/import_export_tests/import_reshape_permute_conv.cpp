// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "import_export_tests/import_reshape_permute_conv.hpp"

#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

void ImportReshapePermuteConv::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, targetDevice, exportConfiguration, importConfiguration, applicationHeader) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, { {1, 336} });

    std::vector<size_t> outFormShapes1 = { 1, 1, 168, 2 };
    auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, outFormShapes1);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

    auto permute1 = std::make_shared<ngraph::opset1::Transpose>(reshape1,
                                                                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));

    auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, { 1, 8 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                                                  ngraph::op::PadType::VALID, 12);

    auto permute2 = std::make_shared<ngraph::opset1::Transpose>(conv1,
                                                                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));

    std::vector<size_t> outFormShapes2 = { 1, 1932 };
    auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes2);
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2) };
    function = std::make_shared<ngraph::Function>(results, params, "ExportImportNetwork");
}

TEST_P(ImportReshapePermuteConv, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions
