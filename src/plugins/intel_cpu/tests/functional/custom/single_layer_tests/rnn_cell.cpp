// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/rnn_cell.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using RNNCellCPUParams = typename std::tuple<std::vector<InputShape>,   // Shapes
                                             std::vector<std::string>,  // Activations
                                             float,                     // Clip
                                             ElementType,               // Network precision
                                             CPUSpecificParams,         // CPU specific params
                                             ov::AnyMap                 // Additional config
                                             >;

class RNNCellCPUTest : public testing::WithParamInterface<RNNCellCPUParams>,
                            virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RNNCellCPUParams> &obj) {
        std::vector<InputShape> inputShapes;
        std::vector<std::string> activations;
        float clip = 0.f;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;

        std::tie(inputShapes, activations, clip, netPrecision, cpuParams, additionalConfig) = obj.param;

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
        std::vector<std::string> activations;
        float clip = 0.f;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;

        std::tie(inputShapes, activations, clip, netPrecision, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        targetDevice = ov::test::utils::DEVICE_CPU;

        init_input_shapes(inputShapes);

        const size_t hiddenSize = targetStaticShapes.front()[1][1];
        const size_t inputSize = targetStaticShapes.front()[0][1];

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

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
        std::vector<ov::Shape> WRB = {{hiddenSize, inputSize}, {hiddenSize, hiddenSize}, {hiddenSize}};
        auto rnnCellOp = utils::make_rnn(paramsOuts, WRB, hiddenSize, activations, {}, {}, clip);

        function = makeNgraphFunction(netPrecision, params, rnnCellOp, "RNNCellCPU");
    }
};

TEST_P(RNNCellCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "RNNCell");
}

namespace {
/* CPU PARAMS */
std::vector<ov::AnyMap> additionalConfig = {{ov::hint::inference_precision(ov::element::f32)},
                                            {ov::hint::inference_precision(ov::element::bf16)}};

CPUSpecificParams cpuParams{{nc, nc}, {nc}, {"ref_any"}, "ref_any"};
std::vector<std::vector<std::string>> activations = {{"relu"}, {"sigmoid"}, {"tanh"}};
// oneDNN supports only zero clip
std::vector<float> clip = {0.f};
std::vector<ElementType> netPrecisions = { ElementType::f32 };

const std::vector<std::vector<ov::test::InputShape>> staticShapes = {
    { { {}, { {1, 1} } },   // Static shapes
      { {}, { {1, 1} } } },
    { { {}, { {1, 30} } },  // Static shapes
      { {}, { {1, 10} } } },
    { { {}, { {5, 1} } },   // Static shapes
      { {}, { {5, 1} } } },
    { { {}, { {5, 30} } },  // Static shapes
      { {}, { {5, 10} } } }
};

INSTANTIATE_TEST_SUITE_P(smoke_static, RNNCellCPUTest,
        ::testing::Combine(::testing::ValuesIn(staticShapes),
                           ::testing::ValuesIn(activations),
                           ::testing::ValuesIn(clip),
                           ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(cpuParams),
                           ::testing::ValuesIn(additionalConfig)),
        RNNCellCPUTest::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynamicShapes = {
    { { { {-1}, 1 },                           // Dynamic shape 0
        { {1, 1}, {3, 1}, {5, 1} } },          // Target shapes
      { { {-1}, 1 },                           // Dynamic shape 1
        { {1, 1}, {3, 1}, {5, 1} } } },        // Target shapes
    { { { {1, 10}, 30 },                       // Dynamic shape 0
        { {2, 30}, {5, 30}, {8, 30} } },       // Target shapes
      { { {1, 10}, 10 },                       // Dynamic shape 1
        { {2, 10}, {5, 10}, {8, 10} } } },     // Target shapes
    { { { {1, 10}, -1 },                       // Dynamic shape 0
        { {2, 30}, {5, 30}, {8, 30} } },       // Target shapes
      { { {1, 10}, {1, 11} },                  // Dynamic shape 1
        { {2, 10}, {5, 10}, {8, 10} } } },     // Target shapes
    { { { {1, 10}, -1 },                       // Dynamic shape 0
        { {2, 30}, {5, 30}, {2, 30}, {8, 30}, {5, 30}, {8, 30} } },  // Target shapes
      { { {1, 10}, {1, 11} },                  // Dynamic shape 1
        { {2, 10}, {5, 10}, {2, 10}, {8, 10}, {5, 10}, {8, 10} } } } // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic, RNNCellCPUTest,
        ::testing::Combine(::testing::ValuesIn(dynamicShapes),
                           ::testing::ValuesIn(activations),
                           ::testing::ValuesIn(clip),
                           ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(cpuParams),
                           ::testing::ValuesIn(additionalConfig)),
        RNNCellCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
