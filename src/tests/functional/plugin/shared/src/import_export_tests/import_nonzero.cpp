// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "import_export_tests/import_nonzero.hpp"

#include "ngraph/opsets/opset5.hpp"

namespace LayerTestsDefinitions {

void ImportNonZero::SetUp() {
    InferenceEngine::Precision netPrecision;
    ngraph::Shape inputShape;
    std::tie(inputShape, netPrecision, targetDevice, exportConfiguration, importConfiguration, applicationHeader) = this->GetParam();
    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto parameter = std::make_shared<ngraph::opset5::Parameter>(ngPrc, inputShape);
    const auto nonZero = std::make_shared<ngraph::opset5::NonZero>(parameter);

    function = std::make_shared<ngraph::Function>(nonZero->outputs(), ngraph::ParameterVector{parameter}, "ExportImportNetwork");
}

TEST_P(ImportNonZero, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions
