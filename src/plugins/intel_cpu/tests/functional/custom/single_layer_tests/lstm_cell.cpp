// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/lstm_cell.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using LSTMCellCpuSpecificParams = typename std::tuple<
        std::vector<InputShape>,           // Shapes
        bool,                              // using decompose to sub-ops transformation
        std::vector<std::string>,          // activations
        float,                             // clip
        ElementType,                       // Network precision
        CPUSpecificParams,                 // CPU specific params
        ov::AnyMap // Additional config
>;

class LSTMCellLayerCPUTest : public testing::WithParamInterface<LSTMCellCpuSpecificParams>,
                             virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMCellCpuSpecificParams>& obj) {
        std::vector<InputShape> inputShapes;
        bool decompose;
        std::vector<std::string> activations;
        float clip = 0.f;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;

        std::tie(inputShapes, decompose, activations, clip, netPrecision, cpuParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < inputShapes.size(); j++) {
                result << ov::test::utils::vec2str(inputShapes[j].second[i]) << (j < inputShapes.size() - 1 ? "_" : "");
            }
            result << "}_";
        }
        result << "decompose=" << decompose << "_";
        result << "activations=" << ov::test::utils::vec2str(activations)  << "_";
        result << "clip=" << clip << "_";
        result << "netPrec=" << netPrecision << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }
        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        bool decompose;
        std::vector<std::string> activations;
        float clip = 0.f;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;
        abs_threshold = 0.05;

        std::tie(inputShapes, decompose, activations, clip, netPrecision, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        targetDevice = ov::test::utils::DEVICE_CPU;

        init_input_shapes(inputShapes);

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        const size_t hiddenSize = targetStaticShapes.front()[1][1];
        const size_t inputSize = targetStaticShapes.front()[0][1];

        auto it = additionalConfig.find(ov::hint::inference_precision.name());
        if (it != additionalConfig.end() && it->second.as<ov::element::Type>() == ov::element::bf16) {
            selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
        } else {
            selectedType = makeSelectedTypeStr(selectedType, netPrecision);
        }

        ov::ParameterVector params;
        ov::OutputVector paramsOuts;
        for (auto&& shape : inputDynamicShapes) {
            auto param = std::make_shared<ov::op::v0::Parameter>(netPrecision, shape);
            params.push_back(param);
            paramsOuts.push_back(param);
        }

        std::vector<ov::Shape> WRB = {{4 * hiddenSize, inputSize}, {4 * hiddenSize, hiddenSize}, {4 * hiddenSize}};
        auto lstmCellOp = utils::make_lstm(paramsOuts, WRB, hiddenSize, activations, {}, {}, clip);

        function = makeNgraphFunction(netPrecision, params, lstmCellOp, "LSTMCell");
    }
};

TEST_P(LSTMCellLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "RNNCell");
}

namespace {
/* CPU PARAMS */
std::vector<ov::AnyMap> additionalConfig = {{{ov::hint::inference_precision(ov::element::bf16)}},
                                            {{ov::hint::inference_precision(ov::element::f32)}}};

CPUSpecificParams cpuParams{{nc, nc, nc}, {nc}, {"ref_any"}, "ref_any"};

std::vector<bool> should_decompose{false};
// oneDNN supports only sigmoid-tanh-tanh
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"}};
// oneDNN supports only zero clip
std::vector<float> clip{0.f};
std::vector<ElementType> netPrecisions = { ElementType::f32 };

const std::vector<std::vector<ov::test::InputShape>> staticShapes = {
    { { {}, { {1, 1} } }, // Static shapes
      { {}, { {1, 1} } },
      { {}, { {1, 1} } } },
    { { {}, { {1, 30} } }, // Static shapes
      { {}, { {1, 10} } },
      { {}, { {1, 10} } } },
    { { {}, { {5, 1} } }, // Static shapes
      { {}, { {5, 1} } },
      { {}, { {5, 1} } } },
    { { {}, { {5, 30} } }, // Static shapes
      { {}, { {5, 10} } },
      { {}, { {5, 10} } } }
};

INSTANTIATE_TEST_SUITE_P(smoke_static, LSTMCellLayerCPUTest,
                ::testing::Combine(::testing::ValuesIn(staticShapes),
                                   ::testing::ValuesIn(should_decompose),
                                   ::testing::ValuesIn(activations),
                                   ::testing::ValuesIn(clip),
                                   ::testing::ValuesIn(netPrecisions),
                                   ::testing::Values(cpuParams),
                                   ::testing::ValuesIn(additionalConfig)),
                LSTMCellLayerCPUTest::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynamicShapes = {
    { { { -1, 1 },                         // Dynamic shape 0
        { {1, 1}, {3, 1}, {5, 1} } },      // Target shapes
      { { -1, 1 },                         // Dynamic shape 1
        { {1, 1}, {3, 1}, {5, 1} } },      // Target shapes
      { { -1, 1 },                         // Dynamic shape 2
        { {1, 1}, {3, 1}, {5, 1} } } },    // Target shapes
    { { { -1, 1 },                         // Dynamic shape 0
        { {1, 1}, {5, 1} } },              // Target shapes
      { { {1, 5}, 1 },                     // Dynamic shape 1
        { {1, 1}, {5, 1} } },              // Target shapes
      { { {1, 5}, 1 },                     // Dynamic shape 2
        { {1, 1}, {5, 1} } } },            // Target shapes
    { { { {1, 20}, 30 },                   // Dynamic shape 0
        { {2, 30}, {5, 30}, {8, 30} } },   // Target shapes
      { { {1, 20}, 10 },                   // Dynamic shape 1
        { {2, 10}, {5, 10}, {8, 10} } },   // Target shapes
      { { {1, 20}, 10 },                   // Dynamic shape 2
        { {2, 10}, {5, 10}, {8, 10} } } }, // Target shapes
    { { { {1, 20}, {28, 32} },             // Dynamic shape 0
        { {2, 30}, {5, 30}, {8, 30} } },   // Target shapes
      { { {1, 20}, {8, 12} },              // Dynamic shape 1
        { {2, 10}, {5, 10}, {8, 10} } },   // Target shapes
      { { {1, 20}, -1 },                   // Dynamic shape 2
        { {2, 10}, {5, 10}, {8, 10} } } }, // Target shapes
    { { { {1, 20}, {28, 32} },             // Dynamic shape 0
        { {2, 30}, {5, 30}, {8, 30}, {2, 30}, {5, 30}, {8, 30} } },   // Target shapes
      { { {1, 20}, {8, 12} },              // Dynamic shape 1
        { {2, 10}, {5, 10}, {8, 10}, {2, 10}, {5, 10}, {8, 10} } },   // Target shapes
      { { {1, 20}, -1 },                   // Dynamic shape 2
        { {2, 10}, {5, 10}, {8, 10}, {2, 10}, {5, 10}, {8, 10} } } }, // Target shapes
    { { { -1, -1 },                         // Dynamic shape 0
        { {37, 512}, {15, 512} } },         // Target shapes
      { { -1, 128 },                        // Dynamic shape 1
        { {37, 128}, {15, 128} } },         // Target shapes
      { { -1, 128 },                        // Dynamic shape 2
        { {37, 128}, {15, 128} } } },       // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic, LSTMCellLayerCPUTest,
                ::testing::Combine(::testing::ValuesIn(dynamicShapes),
                                   ::testing::ValuesIn(should_decompose),
                                   ::testing::ValuesIn(activations),
                                   ::testing::ValuesIn(clip),
                                   ::testing::ValuesIn(netPrecisions),
                                   ::testing::Values(cpuParams),
                                   ::testing::ValuesIn(additionalConfig)),
                LSTMCellLayerCPUTest::getTestCaseName);
} // namespace
}  // namespace test
}  // namespace ov
