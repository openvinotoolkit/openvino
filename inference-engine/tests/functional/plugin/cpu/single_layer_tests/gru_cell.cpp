// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/gru_cell.hpp"
#include <shared_test_classes/single_layer/gru_cell.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "transformations/op_conversions/gru_cell_decomposition.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using GRUCellCpuSpecificParams = typename std::tuple<LayerTestsDefinitions::GRUCellParams, CPUSpecificParams, std::map<std::string, std::string>>;

class GRUCellCPUTest : public testing::WithParamInterface<GRUCellCpuSpecificParams>,
                            virtual public LayerTestsUtils::LayerTestsCommon,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GRUCellCpuSpecificParams> &obj) {
        CPUSpecificParams cpuParams;
        LayerTestsDefinitions::GRUCellParams basicParamsSet;
        std::map<std::string, std::string> additionalConfig;

        std::tie(basicParamsSet, cpuParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::GRUCellTest::getTestCaseName(
            testing::TestParamInfo<LayerTestsDefinitions::GRUCellParams>(basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto &item : additionalConfig) {
                if (item.second == PluginConfigParams::YES)
                    result << "_" << item.first << "=" << item.second;
            }
        }
        return result.str();
    }

protected:
    void SetUp() {
        CPUSpecificParams cpuParams;
        LayerTestsDefinitions::GRUCellParams basicParamsSet;
        std::map<std::string, std::string> additionalConfig;

        bool should_decompose;
        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        bool linear_before_reset;
        InferenceEngine::Precision netPrecision;

        std::tie(basicParamsSet, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(should_decompose, batch, hidden_size, input_size, activations, clip, linear_before_reset, netPrecision, targetDevice) = basicParamsSet;

        std::vector<std::vector<size_t>> inputShapes = {
            {{batch, input_size},
             {batch, hidden_size},
             {3 * hidden_size, input_size},
             {3 * hidden_size, hidden_size},
             {(linear_before_reset ? 4 : 3) * hidden_size}},
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
        auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
        std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
        auto gru_cell = ngraph::builder::makeGRU(
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(params)), WRB, hidden_size, activations, {}, {}, clip, linear_before_reset);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_cell->output(0))};

        function = makeNgraphFunction(ngPrc, params, gru_cell, "gru_cell");
    }
};

TEST_P(GRUCellCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "RNNCell");
}

namespace {
/* CPU PARAMS */
std::vector<std::map<std::string, std::string>> additionalConfig
    = {{{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO}},
       {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}}};

CPUSpecificParams cpuParams{{nc, nc}, {nc}, {"ref_any"}, "ref_any"};

std::vector<bool> should_decompose{false};
std::vector<size_t> batch{1, 5};
std::vector<size_t> hidden_size{1, 10};
std::vector<size_t> input_size{1, 30};
// oneDNN supports only sigmoid-tanh
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh"}};
// oneDNN supports only zero clip
std::vector<float> clip = {0.f};
std::vector<bool> linear_before_reset = {true, false};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

INSTANTIATE_TEST_CASE_P(smoke_GRUCellCPU,
                        GRUCellCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(should_decompose),
                                                              ::testing::ValuesIn(batch),
                                                              ::testing::ValuesIn(hidden_size),
                                                              ::testing::ValuesIn(input_size),
                                                              ::testing::ValuesIn(activations),
                                                              ::testing::ValuesIn(clip),
                                                              ::testing::ValuesIn(linear_before_reset),
                                                              ::testing::ValuesIn(netPrecisions),
                                                              ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                           ::testing::Values(cpuParams),
                                           ::testing::ValuesIn(additionalConfig)),
                        GRUCellCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
