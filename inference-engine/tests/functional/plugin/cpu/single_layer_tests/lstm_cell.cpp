// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/lstm_cell.hpp"
#include <shared_test_classes/single_layer/lstm_cell.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using LSTMCellCpuSpecificParams = typename std::tuple<LayerTestsDefinitions::LSTMCellParams, CPUSpecificParams, std::map<std::string, std::string>>;

class LSTMCellLayerCPUTest : public testing::WithParamInterface<LSTMCellCpuSpecificParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon,
                             public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMCellCpuSpecificParams>& obj) {
        CPUSpecificParams cpuParams;
        LayerTestsDefinitions::LSTMCellParams basicParamsSet;
        std::map<std::string, std::string> additionalConfig;

        std::tie(basicParamsSet, cpuParams, additionalConfig) = obj.param;
        std::ostringstream result;

        result << LayerTestsDefinitions::LSTMCellTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::LSTMCellParams>(
                basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                if (item.second == PluginConfigParams::YES)
                    result << "_" << item.first << "=" << item.second;
            }
        }
        return result.str();
    }

protected:
    void SetUp() override {
        LayerTestsDefinitions::LSTMCellParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        bool should_decompose;
        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        InferenceEngine::Precision netPrecision;
        threshold = 0.05;

        std::tie(basicParamsSet, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(should_decompose, batch, hidden_size, input_size, activations, clip, netPrecision, targetDevice) = basicParamsSet;

        std::vector<std::vector<size_t>> inputShapes = {
            {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size}, {4 * hidden_size, hidden_size}, {4 * hidden_size}},
        };

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        if (additionalConfig[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES) {
            inPrc = outPrc = Precision::BF16;
        } else {
            inPrc = outPrc = netPrecision;
        }

        selectedType += "_";
        selectedType += outPrc.name();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(Precision::FP32);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1], inputShapes[2]});
        std::vector<ngraph::Shape> WRB = {inputShapes[3], inputShapes[4], inputShapes[5]};

        auto lstm_cell = ngraph::builder::makeLSTM(
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(params)), WRB, hidden_size, activations, {}, {}, clip);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(lstm_cell->output(0)),
                                     std::make_shared<ngraph::opset1::Result>(lstm_cell->output(1))};

        function = makeNgraphFunction(ngPrc, params, lstm_cell, "lstm_cell");
    }
};

TEST_P(LSTMCellLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "RNNCell");
}

namespace {
/* CPU PARAMS */
std::vector<std::map<std::string, std::string>> additionalConfig
    = {{{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}},
       {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO}}};

CPUSpecificParams cpuParams{{nc, nc, nc}, {nc}, {"ref_any"}, "ref_any"};

std::vector<bool> should_decompose{false};
std::vector<size_t> batch{5};
std::vector<size_t> hidden_size{1, 10};
std::vector<size_t> input_size{1, 30};
// oneDNN supports only sigmoid-tanh-tanh
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"}};
// oneDNN supports only zero clip
std::vector<float> clip{0.f};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32, InferenceEngine::Precision::BF16};

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCellCPU,
                        LSTMCellLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(should_decompose),
                                                              ::testing::ValuesIn(batch),
                                                              ::testing::ValuesIn(hidden_size),
                                                              ::testing::ValuesIn(input_size),
                                                              ::testing::ValuesIn(activations),
                                                              ::testing::ValuesIn(clip),
                                                              ::testing::ValuesIn(netPrecisions),
                                                              ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                           ::testing::Values(cpuParams),
                                           ::testing::ValuesIn(additionalConfig)),
                        LSTMCellLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
