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
        int64_t,                           // Axis
        int64_t,                           // Batch dims
        ElementType,                       // Network precision
        bool,                              // Is axis input constant
        CPUSpecificParams,                 // CPU specific params
        std::map<std::string, std::string> // Additional config
> GatherLayerTestCPUParams;

class GatherLayerTestCPU : public testing::WithParamInterface<GatherLayerTestCPUParams>,
                           virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherLayerTestCPUParams> obj) {
        std::vector<InputShape> inputShapes;
        int64_t axis, batchDims;
        ElementType netPrecision;
        bool isAxisConstant;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, axis, batchDims, netPrecision, isAxisConstant, cpuParams, additionalConfig) = obj.param;

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
        result << "axis=" << axis << "_";
        result << "batchDims=" << batchDims << "_";
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
        int64_t batchDims;
        ElementType netPrecision;
        bool isAxisConstant = true;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        const ElementType intInputsPrecision = ElementType::i64;

        std::tie(inputShapes, axis, batchDims, netPrecision, isAxisConstant, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes(inputShapes);
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        selectedType = makeSelectedTypeStr(selectedType, netPrecision);

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
                tensor = ov::test::utils::create_and_fill_tensor(
                        funcInput.get_element_type(), targetInputStaticShapes[0], 256, 0, 1);
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

// 1D
const std::vector<std::vector<ov::test::InputShape>> staticInputShapes1D = {
    { { {}, { {4} } },   // Static shapes
      { {}, { {2, 3, 4} } }
    },
    { { {}, { {4} } },   // Static shapes
      { {}, { {1} } },
    },
    { { {}, { {4} } },   // Static shapes
      { {}, { {9} } }
    },
    { { {}, { {5} } },   // Static shapes
      { {}, { {5} } }
    }
};
const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes1D = {
    { { { ov::Dimension{1, 70} },                                                             // Dynamic shape 0
        { {3}, {3}, {5}, {5}, {7}, {7}, {8}, {10}, {16}, {17}, {32}, {55}, {32}, {55} } },    // Target shapes
      { { -1 },                                                                               // Dynamic shape 1
        { {1}, {3}, {4}, {5}, {7}, {8}, {9}, {15}, {16}, {17}, {32}, {55}, {64}, {67} } } }   // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_static_1D, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(staticInputShapes1D),
                    ::testing::Values(0),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_1D, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapes1D),
                    ::testing::Values(0),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true, false),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

// 2D
const std::vector<std::vector<ov::test::InputShape>> staticInputShapes2Dref_ax0 = {
    { { {}, { {3, 33} } },    // Static shapes
      { {}, { {5, 4} } }
    },
    { { {}, { {2, 8} } },    // Static shapes
      { {}, { {3, 15} } },
    },
    { { {}, { {2, 15} } },   // Static shapes
      { {}, { {3, 7} } }
    },
    { { {}, { {1, 17} } },    // Static shapes
      { {}, { {5, 10} } }
    },
    { { {}, { {7, 9} } },     // Static shapes
      { {}, { {8, 9} } }
    }
};
const std::vector<std::vector<ov::test::InputShape>> staticInputShapes2DjitA2_ax0 = {
    { { {}, { {3, 3} } },    // Static shapes
      { {}, { {5, 4} } }
    },
    { { {}, { {2, 4} } },    // Static shapes
      { {}, { {3, 15} } },
    },
    { { {}, { {2, 6} } },   // Static shapes
      { {}, { {3, 7} } }
    },
    { { {}, { {1, 7} } },    // Static shapes
      { {}, { {5, 10} } }
    }
};
const std::vector<std::vector<ov::test::InputShape>> staticInputShapes2DjitA5_ax0 = {
    { { {}, { {3, 3} } },    // Static shapes
      { {}, { {5, 4} } }
    },
    { { {}, { {2, 7} } },    // Static shapes
      { {}, { {3, 15} } },
    },
    { { {}, { {2, 9} } },   // Static shapes
      { {}, { {3, 7} } }
    },
    { { {}, { {1, 15} } },    // Static shapes
      { {}, { {5, 10} } }
    }
};

std::vector<std::vector<ov::test::InputShape>> getShapes2Dax0() {
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        return staticInputShapes2DjitA5_ax0;
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        return staticInputShapes2DjitA2_ax0;
    } else {
        return staticInputShapes2Dref_ax0;
    }
}

const std::vector<std::vector<ov::test::InputShape>> staticInputShapes2Djit_ax1 = {
    { { {}, { {4, 7} } },    // Static shapes
      { {}, { {4, 55} } }
    },
    { { {}, { {4, 17} } },   // Static shapes
      { {}, { {4, 17} } },
    },
    { { {}, { {4, 55} } },   // Static shapes
      { {}, { {4, 7} } }
    },
    { { {}, { {5, 5} } },    // Static shapes
      { {}, { {5, 33} } }
    },
    { { {}, { {7, 7} } },    // Static shapes
      { {}, { {7, 9} } }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_static_2Dref_ax0, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(staticInputShapes2Dref_ax0),
                    ::testing::Values(0),
                    ::testing::Values(0),
                    ::testing::ValuesIn({ElementType::f32}),
                    ::testing::Values(true),
                    ::testing::Values(cpuParamsRef),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_2Djit_ax0, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(getShapes2Dax0()),
                    ::testing::Values(0),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_2Djit_ax1, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(staticInputShapes2Djit_ax1),
                    ::testing::Values(1),
                    ::testing::ValuesIn(std::vector<int64_t>{0, 1}),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes2Djit_ax1 = {
    { { { ov::Dimension{1, 128}, ov::Dimension(3, 80) },          // Dynamic shape 0
        { {3, 4}, {5, 5}, {128, 8}, {7, 5}, {7, 7}, {8, 5} } },    // Target shapes
      { { -1, ov::Dimension(3, 99) },                            // Dynamic shape 1
        { {3, 6}, {5, 4}, {128, 7}, {7, 8}, {7, 9}, {8, 9} } } }   // Target shapes
};
const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes2Dref_ax0 = {
    { { { ov::Dimension{1, 70}, ov::Dimension(3, 80) },          // Dynamic shape 0
        { {4, 7}, {3, 21}, {2, 35}, {1, 41}, {5, 45} } },        // Target shapes
      { { -1, ov::Dimension(3, 99) },                            // Dynamic shape 1
        { {4, 55}, {5, 7}, {8, 35}, {8, 41}, {8, 45} } } }       // Target shapes
};
const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes2DjitA5_ax0 = {
    { { { ov::Dimension{2, 70}, ov::Dimension(1, 80) },          // Dynamic shape 0
        { {2, 1}, {3, 3}, {4, 7}, {5, 9}, {8, 15} } },        // Target shapes
      { { -1, ov::Dimension(3, 99) },                            // Dynamic shape 1
        { {4, 55}, {5, 7}, {8, 35}, {8, 41}, {8, 45} } } }       // Target shapes
};
const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes2DjitA2_ax0 = {
    { { { ov::Dimension{2, 70}, ov::Dimension(1, 80) },          // Dynamic shape 0
        { {2, 1}, {3, 1}, {4, 1}, {5, 1}, {8, 1} } },        // Target shapes
      { { -1, ov::Dimension(3, 99) },                            // Dynamic shape 1
        { {4, 55}, {5, 7}, {8, 35}, {8, 41}, {8, 45} } } }       // Target shapes
};

std::vector<std::vector<ov::test::InputShape>> getShapes2DDynAx0() {
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        return dynamicInputShapes2DjitA5_ax0;
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        return dynamicInputShapes2DjitA2_ax0;
    } else {
        return staticInputShapes2Dref_ax0;
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_2Djit_ax1, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapes2Djit_ax1),
                    ::testing::Values(1),
                    ::testing::ValuesIn(std::vector<int64_t>{1}),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true, false),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_2Dref_ax0, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapes2Dref_ax0),
                    ::testing::Values(0),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true, false),
                    ::testing::Values(cpuParamsRef),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_2Djit_ax0, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(getShapes2DDynAx0()),
                    ::testing::Values(0),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true, false),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

// 4D
const std::vector<std::vector<ov::test::InputShape>> staticInputShapes4Dref = {
    { { {}, { {4, 5, 6, 17} } },    // Static shapes
      { {}, { {2, 5, 1} } }
    },
    { { {}, { {10, 5, 6, 33} } },   // Static shapes
      { {}, { {2, 5, 2} } },
    },
    { { {}, { {16, 5, 6, 65} } },   // Static shapes
      { {}, { {3, 5, 3} } }
    }
};
const std::vector<std::vector<ov::test::InputShape>> staticInputShapes4Djit = {
    { { {}, { {4, 5, 6, 7} } },    // Static shapes
      { {}, { {2, 5, 1} } }
    },
    { { {}, { {10, 5, 6, 7} } },   // Static shapes
      { {}, { {2, 5, 2} } },
    },
    { { {}, { {16, 5, 6, 7} } },   // Static shapes
      { {}, { {3, 5, 3} } }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_static_4Dref, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(staticInputShapes4Dref),
                    ::testing::ValuesIn(std::vector<int64_t>{0, 1, 2}),
                    ::testing::Values(0),
                    ::testing::ValuesIn({ElementType::f32}),
                    ::testing::Values(true),
                    ::testing::Values(cpuParamsRef),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4Djit, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(staticInputShapes4Djit),
                    ::testing::ValuesIn(std::vector<int64_t>{2, -1}),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes4Dref = {
    { { { ov::Dimension(4, 20), 5, 6, 7 },                   // Dynamic shape 0
        { {4, 5, 6, 7}, {10, 5, 6, 7}, {16, 5, 6, 7} } },    // Target shapes
      { { ov::Dimension(2, 4), 5, ov::Dimension(1, 4) },     // Dynamic shape 1
        { {2, 5, 1}, {2, 5, 2}, {3, 5, 3} } } },             // Target shapes
    { { { -1, -1, -1, -1 },                                  // Dynamic shape 0
        { {4, 5, 6, 4}, {10, 5, 6, 8} } },                   // Target shapes
      { { -1, -1, -1 },                                      // Dynamic shape 1
        { {2, 5, 16}, {2, 5, 24} } } }                       // Target shapes
};
const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes4Djit = {
    { { { ov::Dimension(4, 20), 5, -1, 1 },                   // Dynamic shape 0
        { {4, 5, 6, 1}, {10, 5, 15, 1}, {16, 5, 31, 1} } },    // Target shapes
      { { ov::Dimension(2, 4), 5, ov::Dimension(1, 4) },     // Dynamic shape 1
        { {2, 5, 1}, {2, 5, 2}, {3, 5, 3} } } },             // Target shapes
    { { { -1, -1, -1, -1 },                                  // Dynamic shape 0
        { {4, 5, 6, 1}, {10, 5, 9, 1} } },                   // Target shapes
      { { -1, -1, -1 },                                      // Dynamic shape 1
        { {2, 5, 16}, {2, 5, 24} } } }                       // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4Dref, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapes4Dref),
                    ::testing::ValuesIn(std::vector<int64_t>{0, 1, 2}),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true, false),
                    ::testing::Values(cpuParamsRef),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4Djit, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapes4Djit),
                    ::testing::ValuesIn(std::vector<int64_t>{2, -1}),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(true, false),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::ValuesIn(additionalConfig)),
                GatherLayerTestCPU::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
