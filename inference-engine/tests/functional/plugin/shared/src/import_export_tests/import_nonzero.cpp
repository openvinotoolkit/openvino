// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "import_export_tests/import_nonzero.hpp"

#include "ngraph/opsets/opset5.hpp"

namespace LayerTestsDefinitions {

void ImportNonZero::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, targetDevice, exportConfiguration, importConfiguration, applicationHeader) = this->GetParam();
    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto parameter = std::make_shared<ngraph::opset5::Parameter>(ngPrc, ngraph::Shape{1000});
    const auto nonZero = std::make_shared<ngraph::opset5::NonZero>(parameter);

    function = std::make_shared<ngraph::Function>(nonZero->outputs(), ngraph::ParameterVector{parameter}, "ExportImportNetwork");
}

TEST_P(ImportNonZero, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions
