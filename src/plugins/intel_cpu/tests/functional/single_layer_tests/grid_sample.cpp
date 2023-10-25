// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace CPUTestUtils;
using namespace ov::test;
using ov::op::v9::GridSample;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>,                 // Input shapes
        GridSample::InterpolationMode,           // Interpolation mode
        GridSample::PaddingMode,                 // Padding mode
        bool,                                    // Align corners
        ElementType,                             // Data precision
        ElementType,                             // Grid precision
        CPUSpecificParams,                       // CPU specific params
        std::map<std::string, std::string>       // Additional config
> GridSampleLayerTestCPUParams;

class GridSampleLayerTestCPU : public testing::WithParamInterface<GridSampleLayerTestCPUParams>,
                               virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GridSampleLayerTestCPUParams> obj) {
        std::vector<InputShape> inputShapes;
        GridSample::InterpolationMode interpolateMode;
        GridSample::PaddingMode paddingMode;
        bool alignCorners;
        ElementType dataPrecision, gridPrecision;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, interpolateMode, paddingMode, alignCorners, dataPrecision, gridPrecision, cpuParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (size_t i = 0lu; i < inputShapes.size(); i++) {
            result << ov::test::utils::partialShape2str({inputShapes[i].first}) << (i < inputShapes.size() - 1lu ? "_" : "");
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < inputShapes.size(); j++) {
                result << ov::test::utils::vec2str(inputShapes[j].second[i]) << (j < inputShapes.size() - 1lu ? "_" : "");
            }
            result << "}_";
        }

        result << "interpMode=" << (interpolateMode == GridSample::InterpolationMode::BILINEAR ? "BILINEAR" :
                                    interpolateMode == GridSample::InterpolationMode::BICUBIC ? "BICUBIC" : "NEAREST") << "_";
        result << "padMode=" << (paddingMode == GridSample::PaddingMode::ZEROS ? "ZEROS" :
                                 paddingMode == GridSample::PaddingMode::BORDER ? "BORDER" : "REFLECTION") << "_";
        result << "alignCorners=" << (alignCorners ? "True" : "False") << "_";
        result << "dataPrc=" << dataPrecision << "_";
        result << "gridPrc=" << gridPrecision;
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
        abs_threshold = 0.0005;
        std::vector<InputShape> inputShapes;
        GridSample::InterpolationMode interpolateMode;
        GridSample::PaddingMode paddingMode;
        bool alignCorners;
        ElementType dataPrecision, gridPrecision;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, interpolateMode, paddingMode, alignCorners, dataPrecision, gridPrecision, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes(inputShapes);
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        if (additionalConfig[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] == InferenceEngine::PluginConfigParams::YES) {
            selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
        } else {
            auto execType = dataPrecision == ov::element::i32 ? ov::element::i32 : ov::element::f32;
            selectedType = makeSelectedTypeStr(selectedType, execType);
        }
        if (gridPrecision == ov::element::bf16) {
            rel_threshold = 0.01f;
        }

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(dataPrecision, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(gridPrecision, inputDynamicShapes[1])};
        params[0]->set_friendly_name("data");
        params[1]->set_friendly_name("grid");
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));
        GridSample::Attributes attributes = {alignCorners, interpolateMode, paddingMode};
        auto gridSampleNode = std::make_shared<GridSample>(paramOuts[0], paramOuts[1], attributes);

        function = makeNgraphFunction(dataPrecision, params, gridSampleNode, "GridSampleCPU");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;

            if (funcInput.get_node()->get_friendly_name() == "data") {
                int32_t range = std::accumulate(targetInputStaticShapes[0].begin(), targetInputStaticShapes[0].end(), 1u, std::multiplies<uint32_t>());
                tensor = utils::create_and_fill_tensor(
                        funcInput.get_element_type(), targetInputStaticShapes[0], range, -range / 2, 1);
            } else if (funcInput.get_node()->get_friendly_name() == "grid") {
                int32_t range = std::max(targetInputStaticShapes[0][2], targetInputStaticShapes[0][3]) + 2;
                int32_t resolution = range / 2;
                tensor = utils::create_and_fill_tensor(
                        funcInput.get_element_type(), targetInputStaticShapes[1], range, -1, resolution == 0 ? 1 : resolution);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(GridSampleLayerTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "GridSample");
}

namespace {

std::vector<std::map<std::string, std::string>> additionalConfig
    = {{{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO}},
       {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES}}};

std::vector<GridSample::InterpolationMode> interpolateMode {
        GridSample::InterpolationMode::BILINEAR,
        GridSample::InterpolationMode::BICUBIC,
        GridSample::InterpolationMode::NEAREST };

std::vector<GridSample::PaddingMode> paddingMode {
        GridSample::PaddingMode::ZEROS,
        GridSample::PaddingMode::BORDER,
        GridSample::PaddingMode::REFLECTION };

std::vector<bool> alignCorners { true, false };

std::vector<CPUSpecificParams> getCPUInfo() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"});
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx2"}, "jit_avx2"});
    } else if (InferenceEngine::with_cpu_x86_avx()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx"}, "jit_avx"});
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_sse42"}, "jit_sse42"});
    }
    return resCPUParams;
}

std::vector<std::vector<InputShape>> getStaticShapes() {
    // SSE42
    std::vector<std::vector<InputShape>> result = {
        { { {}, { {1, 5, 1, 1} } },   // Static shapes
          { {}, { {1, 1, 1, 2} } }
        },
        { { {}, { {2, 4, 7, 1} } },   // Static shapes
          { {}, { {2, 1, 2, 2} } }
        },
        { { {}, { {3, 3, 3, 3} } },   // Static shapes
          { {}, { {3, 3, 1, 2} } }
        },
        { { {}, { {4, 2, 5, 4} } },   // Static shapes
          { {}, { {4, 2, 2, 2} } }
        },
        { { {}, { {5, 1, 5, 5} } },   // Static shapes
          { {}, { {5, 1, 5, 2} } }
        },
        { { {}, { {4, 2, 4, 6} } },   // Static shapes
          { {}, { {4, 2, 3, 2} } }
        },
        { { {}, { {3, 3, 5, 7} } },   // Static shapes
          { {}, { {3, 7, 1, 2} } }
        },
        { { {}, { {2, 4, 7, 7} } },   // Static shapes
          { {}, { {2, 2, 4, 2} } }
        },
        { { {}, { {2, 5, 8, 8} } },   // Static shapes
          { {}, { {2, 3, 3, 2} } }
        },
        { { {}, { {2, 6, 9, 8} } },   // Static shapes
          { {}, { {2, 2, 5, 2} } }
        }
    };
    // AVX2, AVX
    if (InferenceEngine::with_cpu_x86_avx2() || InferenceEngine::with_cpu_x86_avx()) {
        std::vector<std::vector<InputShape>> tmp = {
            { { {}, { {1, 7, 5, 3} } },   // Static shapes
              { {}, { {1, 1, 11, 2} } }
            },
            { { {}, { {2, 6, 7, 2} } },   // Static shapes
              { {}, { {2, 6, 2, 2} } }
            },
            { { {}, { {3, 5, 6, 3} } },   // Static shapes
              { {}, { {3, 1, 13, 2} } }
            },
            { { {}, { {4, 4, 5, 6} } },   // Static shapes
              { {}, { {4, 2, 7, 2} } }
            },
            { { {}, { {5, 3, 4, 5} } },   // Static shapes
              { {}, { {5, 3, 5, 2} } }
            },
            { { {}, { {4, 2, 7, 6} } },   // Static shapes
              { {}, { {4, 4, 4, 2} } }
            },
            { { {}, { {3, 3, 9, 7} } },   // Static shapes
              { {}, { {3, 1, 17, 2} } }
            },
            { { {}, { {2, 4, 9, 8} } },   // Static shapes
              { {}, { {2, 19, 1, 2} } }
            }
        };
        result.insert(result.end(), tmp.begin(), tmp.end());
    }
    // AVX512
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        std::vector<std::vector<InputShape>> tmp = {
            { { {}, { {1, 7, 2, 9} } },    // Static shapes
              { {}, { {1, 4, 5, 2} } }
            },
            { { {}, { {2, 6, 3, 10} } },   // Static shapes
              { {}, { {2, 3, 7, 2} } },
            },
            { { {}, { {3, 5, 2, 11} } },   // Static shapes
              { {}, { {3, 4, 6, 2} } }
            },
            { { {}, { {4, 4, 4, 12} } },   // Static shapes
              { {}, { {4, 5, 5, 2} } },
            },
            { { {}, { {5, 3, 2, 13} } },   // Static shapes
              { {}, { {5, 1, 31, 2} } },
            },
            { { {}, { {4, 3, 5, 14} } },   // Static shapes
              { {}, { {4, 4, 8, 2} } },
            },
            { { {}, { {3, 2, 2, 15} } },   // Static shapes
              { {}, { {3, 33, 1, 2} } },
            },
            { { {}, { {2, 1, 6, 16} } },   // Static shapes
              { {}, { {2, 8, 8, 2} } },
            },
            { { {}, { {2, 3, 7, 17} } },   // Static shapes
              { {}, { {2, 9, 9, 2} } },
            }
        };
        result.insert(result.end(), tmp.begin(), tmp.end());
    }

    return result;
}

INSTANTIATE_TEST_SUITE_P(smoke_static, GridSampleLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(getStaticShapes()),
                ::testing::ValuesIn(interpolateMode),
                ::testing::ValuesIn(paddingMode),
                ::testing::ValuesIn(alignCorners),
                ::testing::ValuesIn({ElementType::f32, ElementType::i32}),
                ::testing::ValuesIn({ElementType::f32}),
                ::testing::ValuesIn(getCPUInfo()),
                ::testing::Values(additionalConfig[0])),
        GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_static_1, GridSampleLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(getStaticShapes()),
                ::testing::ValuesIn(interpolateMode),
                ::testing::ValuesIn(paddingMode),
                ::testing::ValuesIn(alignCorners),
                ::testing::ValuesIn({ElementType::bf16, ElementType::i8}),
                ::testing::ValuesIn({ElementType::f32, ElementType::bf16}),
                ::testing::ValuesIn(getCPUInfo()),
                ::testing::Values(additionalConfig[0])),
        GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_static_2, GridSampleLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(getStaticShapes()),
                ::testing::ValuesIn(interpolateMode),
                ::testing::ValuesIn(paddingMode),
                ::testing::ValuesIn(alignCorners),
                ::testing::ValuesIn({ElementType::f32}),
                ::testing::ValuesIn({ElementType::bf16}),
                ::testing::ValuesIn(getCPUInfo()),
                ::testing::Values(additionalConfig[0])),
        GridSampleLayerTestCPU::getTestCaseName);

const std::vector<std::vector<InputShape>> dynamicInSapes = {
    { { { ov::Dimension(1, 15), -1, -1, -1 },                               // Dynamic shape 0
        { {1, 1, 1, 1}, {6, 3, 1, 2}, {4, 5, 3, 1}, {2, 7, 2, 2} } },       // Target shapes
      { { ov::Dimension(1, 16), -1, -1, -1 },                               // Dynamic shape 1
        { {1, 1, 1, 2}, {6, 2, 2, 2}, {4, 1, 3, 2}, {2, 1, 2, 2} } } },     // Target shapes
    { { { -1, -1, -1, -1 },                                                 // Dynamic shape 0
        { {1, 2, 1, 5}, {3, 4, 2, 3}, {5, 6, 7, 1}, {7, 8, 2, 4} } },       // Target shapes
      { { -1, -1, -1, 2 },                                                  // Dynamic shape 1
        { {1, 2, 4, 2}, {3, 1, 7, 2}, {5, 2, 3, 2}, {7, 1, 5, 2} } } },     // Target shapes
    { { { ov::Dimension(2, 15), -1, -1, -1 },                               // Dynamic shape 0
        { {8, 3, 3, 3}, {6, 5, 2, 5}, {4, 7, 1, 11}, {2, 9, 3, 4} } },      // Target shapes
      { { -1, 3, 7, 2 },                                                    // Dynamic shape 1
        { {8, 3, 7, 2}, {6, 3, 7, 2}, {4, 3, 7, 2}, {2, 3, 7, 2} } } },     // Target shapes
    { { { 3, 4, 4, 5 },                                                     // Dynamic shape 0
        { {3, 4, 4, 5}, {3, 4, 4, 5}, {3, 4, 4, 5}, {3, 4, 4, 5} } },       // Target shapes
      { { -1, -1, -1, 2 },                                                  // Dynamic shape 1
        { {3, 3, 4, 2}, {3, 1, 11, 2}, {3, 2, 5, 2}, {3, 3, 3, 2} } } },    // Target shapes
    { { { -1, -1, -1, -1 },                                                 // Dynamic shape 0
        { {1, 2, 1, 13}, {3, 4, 7, 2}, {5, 6, 3, 5}, {7, 8, 4, 4} } },      // Target shapes
      { { -1, -1, -1, -1 },                                                 // Dynamic shape 1
        { {1, 4, 4, 2}, {3, 3, 5, 2}, {5, 2, 7, 2}, {7, 1, 13, 2} } } },    // Target shapes
    { { { -1, -1, -1, -1 },                                                 // Dynamic shape 0
        { {2, 11, 1, 17}, {4, 9, 6, 3}, {6, 7, 7, 3}, {8, 3, 2, 11} } },    // Target shapes
      { { -1, -1, -1, 2 },                                                  // Dynamic shape 1
        { {2, 5, 4, 2}, {4, 1, 19, 2}, {6, 6, 3, 2}, {8, 1, 17, 2} } } },   // Target shapes
    { { { 3, -1, -1, -1 },                                                  // Dynamic shape 0
        { {3, 2, 1, 23}, {3, 4, 3, 8}, {3, 6, 5, 5}, {3, 8, 31, 1} } },     // Target shapes
      { { -1, -1, -1, 2 },                                                  // Dynamic shape 1
        { {3, 31, 1, 2}, {3, 6, 4, 2}, {3, 23, 1, 2}, {3, 11, 2, 2} } } },  // Target shapes
    { { { -1, 3, -1, -1 },                                                  // Dynamic shape 0
        { {8, 3, 8, 4}, {6, 3, 33, 1}, {4, 3, 8, 6}, {2, 3, 8, 8} } },      // Target shapes
      { { -1, -1, -1, 2 },                                                  // Dynamic shape 1
        { {8, 8, 8, 2}, {6, 8, 7, 2}, {4, 1, 33, 2}, {2, 4, 8, 2} } } }     // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic, GridSampleLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(dynamicInSapes),
                ::testing::ValuesIn(interpolateMode),
                ::testing::ValuesIn(paddingMode),
                ::testing::ValuesIn(alignCorners),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::ValuesIn(getCPUInfo()),
                ::testing::Values(additionalConfig[0])),
        GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_dynamic, GridSampleLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(dynamicInSapes),
                ::testing::ValuesIn(interpolateMode),
                ::testing::ValuesIn(paddingMode),
                ::testing::ValuesIn(alignCorners),
                ::testing::ValuesIn({ElementType::bf16, ElementType::i32}),
                ::testing::ValuesIn({ElementType::bf16}),
                ::testing::ValuesIn(getCPUInfo()),
                ::testing::Values(additionalConfig[0])),
        GridSampleLayerTestCPU::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
