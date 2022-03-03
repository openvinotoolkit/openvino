// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset5.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<size_t>,                // Input Shape
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::string                         // Application Header
> exportImportNetworkParams;

class ImportNonZero : public testing::WithParamInterface<exportImportNetworkParams>,
                      virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        ngraph::Shape inputShape;
        std::tie(inputShape, netPrecision, targetDevice, applicationHeader) = this->GetParam();
        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const auto parameter = std::make_shared<ngraph::opset5::Parameter>(ngPrc, inputShape);
        const auto nonZero = std::make_shared<ngraph::opset5::NonZero>(parameter);

        function = std::make_shared<ngraph::Function>(nonZero->outputs(), ngraph::ParameterVector{parameter}, "ExportImportNetwork");
        functionRefs = ngraph::clone_function(*function);
    }

    void exportImportNetwork() {
        std::stringstream strm;
        strm.write(applicationHeader.c_str(), applicationHeader.size());
        executableNetwork.Export(strm);

        strm.seekg(0, strm.beg);
        std::string appHeader(applicationHeader.size(), ' ');
        strm.read(&appHeader[0], applicationHeader.size());
        ASSERT_EQ(appHeader, applicationHeader);
        executableNetwork = core->ImportNetwork(strm, targetDevice, configuration);
    }

    void Run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        functionRefs = ngraph::clone_function(*function);
        // load export configuration and save outputs
        LoadNetwork();
        GenerateInputs();
        Infer();
        auto actualOutputs = GetOutputs();

        auto referenceOutputs = CalculateRefs();
        Compare(referenceOutputs, actualOutputs);

        const auto compiledExecNetwork = executableNetwork;
        exportImportNetwork();
        const auto importedExecNetwork = executableNetwork;

        GenerateInputs();
        Infer();

        ASSERT_EQ(importedExecNetwork.GetInputsInfo().size(), compiledExecNetwork.GetInputsInfo().size());
        ASSERT_EQ(importedExecNetwork.GetOutputsInfo().size(), compiledExecNetwork.GetOutputsInfo().size());

        for (const auto& next_input : importedExecNetwork.GetInputsInfo()) {
            ASSERT_NO_THROW(compiledExecNetwork.GetInputsInfo()[next_input.first]);
            Compare(next_input.second->getTensorDesc(), compiledExecNetwork.GetInputsInfo()[next_input.first]->getTensorDesc());
        }
        for (const auto& next_output : importedExecNetwork.GetOutputsInfo()) {
            ASSERT_NO_THROW(compiledExecNetwork.GetOutputsInfo()[next_output.first]);
        }
        auto importedOutputs = GetOutputs();

        ASSERT_EQ(actualOutputs.size(), importedOutputs.size());

        for (size_t i = 0; i < actualOutputs.size(); i++) {
            Compare(actualOutputs[i]->getTensorDesc(), importedOutputs[i]->getTensorDesc());
            Compare(actualOutputs[i], importedOutputs[i]);
        }
    }


    std::string applicationHeader;
};

TEST_P(ImportNonZero, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<std::string> appHeaders = {
        "",
        "APPLICATION_HEADER"
};

std::vector<size_t> inputShape = ngraph::Shape{1000};

INSTANTIATE_TEST_SUITE_P(smoke_ImportNetworkCase, ImportNonZero,
                         ::testing::Combine(
                                 ::testing::Values(inputShape),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                 ::testing::ValuesIn(appHeaders)));

} // namespace
