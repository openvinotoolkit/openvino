// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>,           // Input shapes
        std::tuple<int, int>,              // Axis and Batch dim
        ElementType,                       // Network precision
        bool,                              // Is const Axis
        CPUSpecificParams,                 // CPU specific params
        std::map<std::string, std::string> // Additional config
> GatherLayerTestCPUParams;

class GatherLayerTestCPU : public testing::WithParamInterface<GatherLayerTestCPUParams>,
                           virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherLayerTestCPUParams> obj) {
        std::vector<InputShape> inputShapes;
        std::tuple<int, int> axisAndBatchDims;
        ElementType netPrecision;
        bool isAxisConstant;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, axisAndBatchDims, netPrecision, isAxisConstant, cpuParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (size_t i = 0lu; i < inputShapes.size(); i++) {
            result << CommonTestUtils::partialShape2str({inputShapes[i].first}) << (i < inputShapes.size() - 1lu ? "_" : "");
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < inputShapes.size(); j++) {
                result << CommonTestUtils::vec2str(inputShapes[j].second[i]) << (j < inputShapes.size() - 1lu ? "_" : "");
            }
            result << "}_";
        }
        result << "axis=" << std::get<0>(axisAndBatchDims) << "_";
        result << "batchDims=" << std::get<1>(axisAndBatchDims) << "_";
        result << "netPrc=" << netPrecision << "_";
        result << "constAx=" << (isAxisConstant ? "True" : "False") << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto &item : additionalConfig) {
                if (item.second == InferenceEngine::PluginConfigParams::YES)
                    result << "_" << item.first << "=" << item.second;
            }
        }

        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        std::tuple<int, int> axisAndBatchDims;
        ElementType netPrecision;
        bool isAxisConstant;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        const ElementType intInputsPrecision = ElementType::i64;

        std::tie(inputShapes, axisAndBatchDims, netPrecision, isAxisConstant, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        axis = std::get<0>(axisAndBatchDims);
        const int batchDims = std::get<1>(axisAndBatchDims);
        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes(inputShapes);
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        if (additionalConfig[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] == InferenceEngine::PluginConfigParams::YES) {
            selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
        } else {
            selectedType = makeSelectedTypeStr(selectedType, netPrecision);
        }

        if (!isAxisConstant) {
            inputDynamicShapes.push_back({1});
            for (size_t i = 0lu; i < targetStaticShapes.size(); i++) {
                targetStaticShapes[i].push_back({1});
            }
        }

        ngraph::ParameterVector params {
            std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0]),
            std::make_shared<ov::op::v0::Parameter>(intInputsPrecision, inputDynamicShapes[1])
        };
        params[0]->set_friendly_name("data");
        params[1]->set_friendly_name("indices");
        if (!isAxisConstant) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(intInputsPrecision, inputDynamicShapes[2]));
            params[2]->set_friendly_name("axis");
        }
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));
        std::shared_ptr<ov::Node> gatherNode;
        if (isAxisConstant) {
            gatherNode = std::make_shared<ov::op::v8::Gather>(paramOuts[0], paramOuts[1],
                    ov::op::v0::Constant::create(intInputsPrecision, ov::Shape({1}), { axis }), batchDims);
        } else {
            gatherNode = std::make_shared<ov::op::v8::Gather>(paramOuts[0], paramOuts[1], paramOuts[2], batchDims);
        }

        function = makeNgraphFunction(netPrecision, params, gatherNode, "GatherCPU");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        const auto& funcInputs = function->inputs();
        inputs.clear();

        const size_t normAxis = axis < 0 ? axis + targetInputStaticShapes[0].size() : axis;
        const int32_t axisDim = targetInputStaticShapes[0][normAxis];

        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;

            if (funcInput.get_node()->get_friendly_name() == "data") {
                const auto dataTypeSize = funcInput.get_element_type().size();
                const uint32_t range = dataTypeSize == 4 ? 0x7FFFFFFF : dataTypeSize == 2 ? 0xFFFF : 0xFF;
                tensor = ov::test::utils::create_and_fill_tensor(
                        funcInput.get_element_type(), targetInputStaticShapes[0], range, 0, 1);
            } else if (funcInput.get_node()->get_friendly_name() == "indices") {
                tensor = ov::test::utils::create_and_fill_tensor(
                        funcInput.get_element_type(), targetInputStaticShapes[1], axisDim * 2, -axisDim, 1);
            } else if (funcInput.get_node()->get_friendly_name() == "axis") {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), {1}, 1, axis, 1);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    int64_t axis = 0;
};

TEST_P(GatherLayerTestCPU, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckPluginRelatedResults(executableNetwork, "Gather");
}

namespace {
const std::vector<ElementType> netPrecisions = {
        ElementType::f32,
        ElementType::bf16,
        ElementType::i8
};

std::vector<std::map<std::string, std::string>> additionalConfig
    = {{{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO}},
       {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES}}};

std::vector<bool> isAxisConst{true, false};
const CPUSpecificParams cpuParamsRef{{}, {}, {"ref_any"}, "ref_any"};

std::vector<CPUSpecificParams> getCPUInfo() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"});
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx2"}, "jit_avx2"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

///// 1D /////
const std::vector<std::vector<ov::test::InputShape>> staticInputShapes1D = {
    { { {}, { {1} } }, { {}, { {1} } } },
    { { {}, { {2} } }, { {}, { {2} } } },
    { { {}, { {3} } }, { {}, { {3} } } },
    { { {}, { {4} } }, { {}, { {4} } } },
    { { {}, { {5} } }, { {}, { {5} } } },
    { { {}, { {6} } }, { {}, { {6} } } },
    { { {}, { {7} } }, { {}, { {7} } } },
    { { {}, { {8} } }, { {}, { {8} } } },
    { { {}, { {9} } }, { {}, { {9} } } },
    { { {}, { {11} } }, { {}, { {11} } } },
    { { {}, { {13} } }, { {}, { {13} } } },
    { { {}, { {15} } }, { {}, { {15} } } },
    { { {}, { {16} } }, { {}, { {16} } } },
    { { {}, { {17} } }, { {}, { {17} } } },
    { { {}, { {19} } }, { {}, { {19} } } },
    { { {}, { {23} } }, { {}, { {23} } } },
    { { {}, { {24} } }, { {}, { {24} } } },
    { { {}, { {32} } }, { {}, { {32} } } },
    { { {}, { {33} } }, { {}, { {33} } } },
    { { {}, { {37} } }, { {}, { {37} } } },
    { { {}, { {41} } }, { {}, { {41} } } },
    { { {}, { {48} } }, { {}, { {48} } } },
    { { {}, { {51} } }, { {}, { {51} } } },
    { { {}, { {63} } }, { {}, { {63} } } },
    { { {}, { {64} } }, { {}, { {64} } } },
    { { {}, { {65} } }, { {}, { {65} } } }
};

INSTANTIATE_TEST_SUITE_P(smoke_static_1D, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(staticInputShapes1D),
                    ::testing::Values(std::tuple<int, int>{0, 0}),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::Values(additionalConfig[0])),
                GatherLayerTestCPU::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes1D = {
    { { { ov::Dimension{1, 70} },                                                             // Dynamic shape 0
        { {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {11}, {13}, {15}, {16}, {17}, {19}, {23}, {24}, {32}, {55}, {63}, {64}, {65} } }, // Target shapes
      { { -1 },                                                                               // Dynamic shape 1
        { {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {11}, {13}, {15}, {16}, {17}, {19}, {23}, {24}, {32}, {55}, {63}, {64}, {65} } } } // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_1D, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapes1D),
                    ::testing::Values(std::tuple<int, int>{0, 0}),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true, false),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::Values(additionalConfig[0])),
                GatherLayerTestCPU::getTestCaseName);

///// 4D JIT /////
std::vector<std::vector<ov::test::InputShape>> get4DShapesJitStat() {
    std::vector<std::vector<ov::test::InputShape>> result = {};
    if (InferenceEngine::with_cpu_x86_avx2()) {
        result = {
            { { {}, { {18, 2, 2, 1} } },   // Static shapes
              { {}, { {18, 2, 8} } }
            },
            { { {}, { {17, 2, 2, 2} } },   // Static shapes
              { {}, { {17, 2, 7} } }
            },
            { { {}, { {16, 2, 2, 3} } },   // Static shapes
              { {}, { {16, 2, 6} } }
            },
            { { {}, { {15, 2, 2, 4} } },   // Static shapes
              { {}, { {15, 2, 5} } }
            },
            { { {}, { {14, 2, 2, 5} } },   // Static shapes
              { {}, { {14, 2, 4} } }
            },
            { { {}, { {13, 2, 2, 6} } },   // Static shapes
              { {}, { {13, 2, 3} } }
            },
            { { {}, { {12, 2, 2, 7} } },   // Static shapes
              { {}, { {12, 2, 2} } }
            },
            { { {}, { {11, 2, 2, 8} } },   // Static shapes
              { {}, { {11, 2, 1} } }
            }
        };
    }
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        std::vector<std::vector<ov::test::InputShape>> tmp = {
            { { {}, { {19, 4, 2, 9} } },    // Static shapes
              { {}, { {19, 4, 16} } }
            },
            { { {}, { {20, 4, 2, 10} } },   // Static shapes
              { {}, { {20, 4, 15} } },
            },
            { { {}, { {21, 4, 2, 11} } },   // Static shapes
              { {}, { {21, 4, 14} } }
            },
            { { {}, { {22, 4, 2, 12} } },   // Static shapes
              { {}, { {22, 4, 13} } },
            },
            { { {}, { {23, 4, 2, 13} } },   // Static shapes
              { {}, { {23, 4, 12} } },
            },
            { { {}, { {24, 4, 2, 14} } },   // Static shapes
              { {}, { {24, 4, 11} } },
            },
            { { {}, { {25, 4, 2, 15} } },   // Static shapes
              { {}, { {25, 4, 10} } },
            },
            { { {}, { {26, 4, 2, 16} } },   // Static shapes
              { {}, { {26, 4, 9} } },
            }
        };
        result.insert(result.end(), tmp.begin(), tmp.end());
    }

    return result;
}

std::vector<std::tuple<int, int>> get4DAxisBatchJitStat(ov::element::Type type) {
    std::vector<std::tuple<int, int>> result = {};
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        if (type.size() == 4 || type.size() == 2 || type.size() == 1)
            return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}, {2, 0}, {2, 1}, {2, 2}};
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        if (type.size() == 4)
            return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}, {2, 0}, {2, 1}, {2, 2}};
        else if (type.size() == 2 || type.size() == 1)
            return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}};
    }
    return {};
}

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit32, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(get4DShapesJitStat()),
                    ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::f32)),
                    ::testing::Values(ElementType::f32),
                    ::testing::Values(true),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit16, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(get4DShapesJitStat()),
                    ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::bf16)),
                    ::testing::Values(ElementType::bf16),
                    ::testing::Values(true),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::Values(additionalConfig[0])),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit8, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(get4DShapesJitStat()),
                    ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::i8)),
                    ::testing::Values(ElementType::i8),
                    ::testing::Values(true),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::Values(additionalConfig[0])),
                GatherLayerTestCPU::getTestCaseName);


std::vector<std::vector<ov::test::InputShape>> get4DShapesJitDyn() {
    std::vector<std::vector<ov::test::InputShape>> result = {};
    if (InferenceEngine::with_cpu_x86_avx2()) {
        result = {
            { { { ov::Dimension(5, 15), -1, -1, -1 },                            // Dynamic shape 0
                { {8, 2, 2, 1}, {10, 2, 2, 2}, {8, 2, 2, 3}, {10, 2, 2, 4}} },   // Target shapes
              { { ov::Dimension(4, 16), -1, -1 },                                // Dynamic shape 1
                { {8, 2, 8}, {10, 2, 7}, {8, 2, 6}, {10, 2, 5} } } },            // Target shapes
            { { { -1, -1, -1, -1 },                                              // Dynamic shape 0
                { {8, 2, 2, 5}, {10, 2, 2, 6}, {8, 2, 2, 7}, {10, 2, 2, 8}} },   // Target shapes
              { { -1, -1, -1 },                                                  // Dynamic shape 1
                { {8, 2, 4}, {10, 2, 3}, {8, 2, 2}, {10, 2, 1} } } },            // Target shapes
            { { { ov::Dimension(5, 15), -1, -1, -1 },                            // Dynamic shape 0
                { {10, 2, 2, 1}, {10, 2, 2, 2}, {10, 2, 2, 3}, {10, 2, 2, 4}} }, // Target shapes
              { { 10, 2, 5 },                                                    // Dynamic shape 1
                { {10, 2, 5}, {10, 2, 5}, {10, 2, 5}, {10, 2, 5} } } },          // Target shapes
            { { { 8, 2, 2, 5 },                                                  // Dynamic shape 0
                { {8, 2, 2, 5}, {8, 2, 2, 5}, {8, 2, 2, 5}, {8, 2, 2, 5}} },     // Target shapes
              { { -1, -1, -1 },                                                  // Dynamic shape 1
                { {8, 2, 4}, {8, 2, 3}, {8, 2, 2}, {8, 2, 1} } } }               // Target shapes
        };
    }
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        std::vector<std::vector<ov::test::InputShape>> tmp = {
            { { { ov::Dimension(5, 15), -1, -1, -1 },                               // Dynamic shape 0
                { {8, 2, 2, 9}, {10, 2, 2, 10}, {8, 2, 2, 11}, {10, 2, 2, 12}} },   // Target shapes
              { { ov::Dimension(4, 16), -1, -1 },                                   // Dynamic shape 1
                { {8, 2, 16}, {10, 2, 15}, {8, 2, 14}, {10, 2, 13} } } },           // Target shapes
            { { { -1, -1, -1, -1 },                                                 // Dynamic shape 0
                { {8, 2, 2, 13}, {10, 2, 2, 14}, {8, 2, 2, 15}, {10, 2, 2, 16}} },  // Target shapes
              { { -1, -1, -1 },                                                     // Dynamic shape 1
                { {8, 2, 12}, {10, 2, 11}, {8, 2, 10}, {10, 2, 9} } } },            // Target shapes
            { { { ov::Dimension(5, 15), -1, -1, -1 },                               // Dynamic shape 0
                { {10, 2, 2, 9}, {10, 2, 2, 10}, {10, 2, 2, 11}, {10, 2, 2, 12}} }, // Target shapes
              { { 10, 2, 16 },                                                       // Dynamic shape 1
                { {10, 2, 16}, {10, 2, 16}, {10, 2, 16}, {10, 2, 16} } } },         // Target shapes
            { { { 8, 2, 2, 15 },                                                    // Dynamic shape 0
                { {8, 2, 2, 15}, {8, 2, 2, 15}, {8, 2, 2, 15}, {8, 2, 2, 15}} },    // Target shapes
              { { -1, -1, -1 },                                                     // Dynamic shape 1
                { {8, 2, 12}, {8, 2, 11}, {8, 2, 10}, {8, 2, 9} } } }               // Target shapes
        };
        result.insert(result.end(), tmp.begin(), tmp.end());
    }

    return result;
}

std::vector<std::tuple<int, int>> get4DAxisBatchJitDyn(ov::element::Type type) {
    std::vector<std::tuple<int, int>> result = {};
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        if (type.size() == 4 || type.size() == 2 || type.size() == 1)
            return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}};
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        if (type.size() == 4 || type.size() == 2 || type.size() == 1)
            return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}};
    }
    return {};
}

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit32, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(get4DShapesJitDyn()),
                    ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::f32)),
                    ::testing::Values(ElementType::f32),
                    ::testing::ValuesIn(isAxisConst),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit16, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(get4DShapesJitDyn()),
                    ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::bf16)),
                    ::testing::Values(ElementType::bf16),
                    ::testing::ValuesIn(isAxisConst),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::Values(additionalConfig[0])),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit8, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(get4DShapesJitDyn()),
                    ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::i8)),
                    ::testing::Values(ElementType::i8),
                    ::testing::ValuesIn(isAxisConst),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::Values(additionalConfig[0])),
                GatherLayerTestCPU::getTestCaseName);


///// 4D REFERENCE /////
std::vector<std::vector<ov::test::InputShape>> get4DShapesRefStat() {
    std::vector<std::vector<ov::test::InputShape>> result = {};
    if (InferenceEngine::with_cpu_x86_avx2()) {
        result = {
            { { {}, { {10, 2, 9, 9} } },   // Static shapes
              { {}, { {10, 2, 8} } }
            },
            { { {}, { {11, 2, 9, 2} } },   // Static shapes
              { {}, { {11, 2, 7} } }
            },
            { { {}, { {12, 2, 9, 3} } },   // Static shapes
              { {}, { {12, 2, 6} } }
            },
            { { {}, { {13, 2, 9, 4} } },   // Static shapes
              { {}, { {13, 2, 5} } }
            },
            { { {}, { {14, 2, 9, 5} } },   // Static shapes
              { {}, { {14, 2, 4} } }
            },
            { { {}, { {15, 2, 9, 6} } },   // Static shapes
              { {}, { {15, 2, 3} } }
            },
            { { {}, { {16, 2, 9, 7} } },   // Static shapes
              { {}, { {16, 2, 2} } }
            },
            { { {}, { {17, 2, 9, 8} } },   // Static shapes
              { {}, { {17, 2, 1} } }
            }
        };
    }
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        std::vector<std::vector<ov::test::InputShape>> tmp = {
            { { {}, { {25, 4, 4, 17} } },    // Static shapes
              { {}, { {25, 4, 16} } }
            },
            { { {}, { {24, 4, 4, 18} } },   // Static shapes
              { {}, { {24, 4, 15} } },
            },
            { { {}, { {23, 4, 4, 19} } },   // Static shapes
              { {}, { {23, 4, 14} } }
            },
            { { {}, { {22, 4, 4, 20} } },   // Static shapes
              { {}, { {22, 4, 13} } },
            },
            { { {}, { {21, 4, 4, 21} } },   // Static shapes
              { {}, { {21, 4, 12} } },
            },
            { { {}, { {20, 4, 4, 22} } },   // Static shapes
              { {}, { {20, 4, 11} } },
            },
            { { {}, { {19, 4, 4, 23} } },   // Static shapes
              { {}, { {19, 4, 10} } },
            },
            { { {}, { {18, 4, 4, 24} } },   // Static shapes
              { {}, { {18, 4, 9} } },
            }
        };
        result.insert(result.end(), tmp.begin(), tmp.end());
    }

    return result;
}

std::vector<std::tuple<int, int>> get4DAxisBatchRefStat(ov::element::Type type) {
    std::vector<std::tuple<int, int>> result = {};
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        if (type.size() == 4)
            return std::vector<std::tuple<int, int>>{{1, 0}, {1, 1}, {0, 0}};
        else if (type.size() == 2 || type.size() == 1)
            return std::vector<std::tuple<int, int>>{{0, 0}};
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        if (type.size() == 4)
            return std::vector<std::tuple<int, int>>{{1, 0}, {1, 1}, {0, 0}};
        else if (type.size() == 2 || type.size() == 1)
            return std::vector<std::tuple<int, int>>{{2, 0}, {2, 1}, {2, 2}, {1, 0}, {1, 1}, {0, 0}};
    }
    return {};
}

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref32, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(get4DShapesRefStat()),
                    ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::f32)),
                    ::testing::Values(ElementType::f32),
                    ::testing::Values(true),
                    ::testing::Values(cpuParamsRef),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref16, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(get4DShapesRefStat()),
                    ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::bf16)),
                    ::testing::Values(ElementType::bf16),
                    ::testing::Values(true),
                    ::testing::Values(cpuParamsRef),
                    ::testing::Values(additionalConfig[0])),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref8, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(get4DShapesRefStat()),
                    ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::i8)),
                    ::testing::Values(ElementType::i8),
                    ::testing::Values(true),
                    ::testing::Values(cpuParamsRef),
                    ::testing::Values(additionalConfig[0])),
                GatherLayerTestCPU::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
