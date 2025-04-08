/// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using AdaPoolSpecificParams = std::tuple<std::vector<int>,          // pooled vector
                                         std::vector<InputShape>>;  // feature map shape

using AdaPoolLayerTestParams = std::tuple<AdaPoolSpecificParams,
                                          std::string,    // mode
                                          bool,           // second Input is Constant
                                          ElementType,    // Net precision
                                          TargetDevice>;  // Device name

using AdaPoolLayerCPUTestParamsSet = std::tuple<AdaPoolLayerTestParams, CPUSpecificParams>;

class AdaPoolLayerCPUTest : public testing::WithParamInterface<AdaPoolLayerCPUTestParamsSet>,
                            virtual public SubgraphBaseTest,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AdaPoolLayerCPUTestParamsSet> obj) {
        AdaPoolLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType netPr;
        bool isStatic;
        AdaPoolSpecificParams adaPar;
        std::vector<int> pooledSpatialShape;
        std::vector<InputShape> inputShape;
        std::string mode;
        std::tie(adaPar, mode, isStatic, netPr, td) = basicParamsSet;
        std::tie(pooledSpatialShape, inputShape) = adaPar;
        std::ostringstream result;
        result << "AdaPoolTest_";
        result << "IS=(";
        for (const auto& shape : inputShape) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShape) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "OS=" << ov::test::utils::vec2str(pooledSpatialShape) << "(spat.)_";
        result << netPr << "_";
        result << mode << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams) << "_";
        result << std::to_string(obj.index);
        return result.str();
    }

protected:
    void SetUp() override {
        AdaPoolLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        AdaPoolSpecificParams adaPoolParams;
        ElementType netPrecision;
        bool isStatic;
        std::vector<InputShape> inputShape;
        std::tie(adaPoolParams, mode, isStatic, netPrecision, targetDevice) = basicParamsSet;
        std::tie(pooledVector, inputShape) = adaPoolParams;

        init_input_shapes(inputShape);
        if (!isStatic) {
            for (auto& target : targetStaticShapes) {
                target.push_back({pooledVector.size()});
            }
        }

        selectedType = std::string("unknown_FP32");
        if (netPrecision == ElementType::bf16) {
            rel_threshold = 1e-2;
        }
        function = createFunction(isStatic);
        if (function->get_parameters().size() == 2) {
            generatePooledVector();
            functionRefs = createFunction(true);
        }
    }

    void generatePooledVector() {
        std::random_device rd;
        std::uniform_int_distribution<int32_t> distribution(1, 5);
        for (size_t i = 0; i < pooledVector.size(); i++) {
            pooledVector[i] = distribution(rd);
        }
    }

    std::shared_ptr<ov::Model> createFunction(bool secondInputConst) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0])};
        params.front()->set_friendly_name("ParamsInput");
        std::shared_ptr<ov::Node> secondInput;
        if (secondInputConst) {
            secondInput = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{pooledVector.size()}, pooledVector);
        } else {
            auto pooledParam =
                std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{pooledVector.size()});
            pooledParam->set_friendly_name("ParamSecondInput");
            params.push_back(pooledParam);
            secondInput = pooledParam;
        }

        auto adapoolMax = std::make_shared<ov::op::v8::AdaptiveMaxPool>(params[0], secondInput, ov::element::i32);
        adapoolMax->get_rt_info() = getCPUInfo();
        auto adapoolAvg = std::make_shared<ov::op::v8::AdaptiveAvgPool>(params[0], secondInput);
        adapoolAvg->get_rt_info() = getCPUInfo();

        auto function = (mode == "max" ? std::make_shared<ov::Model>(adapoolMax->outputs(), params, "AdaPoolMax")
                                       : std::make_shared<ov::Model>(adapoolAvg->outputs(), params, "AdaPoolAvg"));
        return function;
    }

    void validate() override {
        auto actualOutputs = get_plugin_outputs();
        if (function->get_parameters().size() == 2) {
            auto pos = std::find_if(inputs.begin(),
                                    inputs.end(),
                                    [](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor>& params) {
                                        return params.first->get_friendly_name() == "ParamSecondInput";
                                    });
            OPENVINO_ASSERT(pos != inputs.end());
            inputs.erase(pos);
        }
        auto expectedOutputs = calculate_refs();
        if (expectedOutputs.empty()) {
            return;
        }
        ASSERT_EQ(actualOutputs.size(), expectedOutputs.size())
            << "model interpreter has " << expectedOutputs.size() << " outputs, while OV " << actualOutputs.size();

        compare(expectedOutputs, actualOutputs);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 1) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                auto* dataPtr = tensor.data<int32_t>();
                for (size_t i = 0; i < pooledVector.size(); i++) {
                    dataPtr[i] = pooledVector[i];
                }
            } else {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = 2560;
                in_data.resolution = 256;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

private:
    std::vector<int> pooledVector;
    std::string mode;
};

TEST_P(AdaPoolLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "AdaptivePooling");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice(std::string dims = "3D", std::string modeStr = "max") {
    std::vector<CPUSpecificParams> resCPUParams;
    if (modeStr == "max") {
        if (dims == "5D") {
            resCPUParams.push_back(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}});  // i.e. two equal output layouts
            resCPUParams.push_back(CPUSpecificParams{{ndhwc, x}, {ndhwc, ncdhw}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nCdhw16c, x}, {nCdhw16c, ncdhw}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x}, {nCdhw8c, ncdhw}, {}, {}});
            }
        } else if (dims == "4D") {
            resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw}, {}, {}});  // i.e. two equal output layouts
            resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nchw}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nChw16c, x}, {nChw16c, nchw}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c, nchw}, {}, {}});
            }
        } else {
            resCPUParams.push_back(CPUSpecificParams{{ncw, x}, {ncw}, {}, {}});  // i.e. two equal output layouts
            resCPUParams.push_back(CPUSpecificParams{{nwc, x}, {nwc, ncw}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nCw16c, x}, {nCw16c, ncw}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nCw8c, x}, {nCw8c, ncw}, {}, {}});
            }
        }
    } else {
        if (dims == "5D") {
            resCPUParams.push_back(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}});
            resCPUParams.push_back(CPUSpecificParams{{ndhwc, x}, {ndhwc}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nCdhw16c, x}, {nCdhw16c}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x}, {nCdhw8c}, {}, {}});
            }
        } else if (dims == "4D") {
            resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw}, {}, {}});
            resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nChw16c, x}, {nChw16c}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c}, {}, {}});
            }
        } else {
            resCPUParams.push_back(CPUSpecificParams{{ncw, x}, {ncw}, {}, {}});
            resCPUParams.push_back(CPUSpecificParams{{nwc, x}, {nwc}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nCw16c, x}, {nCw16c}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nCw8c, x}, {nCw8c}, {}, {}});
            }
        }
    }
    return resCPUParams;
}

const std::vector<ElementType> netPrecisions = {ElementType::f32, ElementType::bf16};

const std::vector<std::vector<int>> pooled3DVector = {{1}, {3}, {5}};
const std::vector<std::vector<int>> pooled4DVector = {{1, 1}, {3, 5}, {5, 5}};

const std::vector<std::vector<int>> pooled5DVector = {
    {1, 1, 1},
    {3, 5, 1},
    {3, 5, 3},
};

std::vector<std::vector<ov::Shape>> staticInput3DShapeVector = {{{1, 17, 3}, {3, 7, 5}}};

const std::vector<std::vector<InputShape>> input3DShapeVector = {
    {{{{-1, 17, -1}, {{1, 17, 3}, {3, 17, 5}, {3, 17, 5}}}},
     {{{{1, 10}, 20, {1, 10}}, {{1, 20, 5}, {2, 20, 4}, {3, 20, 6}}}}}};

std::vector<std::vector<ov::Shape>> staticInput4DShapeVector = {{{1, 3, 1, 1}, {3, 17, 5, 2}}};

const std::vector<std::vector<InputShape>> input4DShapeVector = {
    {{{{-1, 3, -1, -1}, {{1, 3, 1, 1}, {3, 3, 5, 2}, {3, 3, 5, 2}}}},
     {{{{1, 10}, 3, {1, 10}, {1, 10}}, {{2, 3, 10, 6}, {3, 3, 6, 5}, {3, 3, 6, 5}}}}}};

std::vector<std::vector<ov::Shape>> staticInput5DShapeVector = {{{1, 17, 2, 5, 2}, {3, 17, 4, 5, 4}}};

const std::vector<std::vector<InputShape>> input5DShapeVector = {
    {{{{-1, 17, -1, -1, -1}, {{1, 17, 2, 5, 2}, {3, 17, 4, 5, 4}, {3, 17, 4, 5, 4}}}},
     {{{{1, 10}, 3, {1, 10}, {1, 10}, {1, 10}}, {{3, 3, 2, 5, 2}, {1, 3, 4, 5, 4}, {1, 3, 4, 5, 4}}}}}};

const auto adaPool3DParams = ::testing::Combine(::testing::ValuesIn(pooled3DVector),     // output spatial shape
                                                ::testing::ValuesIn(input3DShapeVector)  // feature map shape
);

const auto adaPool4DParams = ::testing::Combine(::testing::ValuesIn(pooled4DVector),     // output spatial shape
                                                ::testing::ValuesIn(input4DShapeVector)  // feature map shape
);

const auto adaPool5DParams = ::testing::Combine(::testing::ValuesIn(pooled5DVector),     // output spatial shape
                                                ::testing::ValuesIn(input5DShapeVector)  // feature map shape
);

const auto staticAdaPool3DParams = ::testing::Combine(
    ::testing::ValuesIn(pooled3DVector),                                                 // output spatial shape
    ::testing::ValuesIn(static_shapes_to_test_representation(staticInput3DShapeVector))  // feature map shape
);

const auto staticAdaPool4DParams = ::testing::Combine(
    ::testing::ValuesIn(pooled4DVector),                                                 // output spatial shape
    ::testing::ValuesIn(static_shapes_to_test_representation(staticInput4DShapeVector))  // feature map shape
);

const auto staticAdaPool5DParams = ::testing::Combine(
    ::testing::ValuesIn(pooled5DVector),                                                 // output spatial shape
    ::testing::ValuesIn(static_shapes_to_test_representation(staticInput5DShapeVector))  // feature map shape
);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolAvg3DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(adaPool3DParams,
                                                               ::testing::Values("avg"),
                                                               ::testing::Values(false),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("3D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolAvg4DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(adaPool4DParams,
                                                               ::testing::Values("avg"),
                                                               ::testing::Values(false),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("4D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolAvg5DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(adaPool5DParams,
                                                               ::testing::Values("avg"),
                                                               ::testing::Values(false),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("5D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolMax3DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(adaPool3DParams,
                                                               ::testing::Values("max"),
                                                               ::testing::Values(false),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("3D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolMax4DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(adaPool4DParams,
                                                               ::testing::Values("max"),
                                                               ::testing::Values(false),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("4D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolMax5DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(adaPool5DParams,
                                                               ::testing::Values("max"),
                                                               ::testing::Values(false),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("5D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolAvg3DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(staticAdaPool3DParams,
                                                               ::testing::Values("avg"),
                                                               ::testing::Values(true),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("3D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolAvg4DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(staticAdaPool4DParams,
                                                               ::testing::Values("avg"),
                                                               ::testing::Values(true),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("4D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolAvg5DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(staticAdaPool5DParams,
                                                               ::testing::Values("avg"),
                                                               ::testing::Values(true),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("5D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolMax3DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(staticAdaPool3DParams,
                                                               ::testing::Values("max"),
                                                               ::testing::Values(true),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("3D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolMax4DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(staticAdaPool4DParams,
                                                               ::testing::Values("max"),
                                                               ::testing::Values(true),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("4D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolMax5DLayoutTest,
                         AdaPoolLayerCPUTest,
                         ::testing::Combine(::testing::Combine(staticAdaPool5DParams,
                                                               ::testing::Values("max"),
                                                               ::testing::Values(true),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice("5D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

// in 1-channel cases  {..., 1, 1, 1} shape cannot be correctly resolved on oneDnn level, so it was removed from
// instances

const std::vector<std::vector<InputShape>> input3DShape1Channel = {
    {{{{-1, -1, -1}, {{1, 1, 2}, {1, 1, 2}, {1, 1, 2}}}},
     {{{{1, 10}, {1, 10}, {1, 10}}, {{1, 1, 2}, {2, 1, 2}, {2, 1, 2}}}}}};

const std::vector<std::vector<InputShape>> input4DShape1Channel = {
    {{{{-1, -1, -1, -1}, {{1, 1, 1, 2}, {2, 1, 2, 1}, {2, 1, 2, 1}}}},
     {{{{1, 10}, {1, 10}, {1, 10}, {1, 10}}, {{1, 1, 1, 2}, {1, 1, 1, 2}, {2, 1, 2, 1}}}}}};

const std::vector<std::vector<InputShape>> input5DShape1Channel = {
    {{{{-1, -1, -1, -1, -1}, {{1, 1, 1, 1, 2}, {1, 1, 1, 1, 2}, {2, 1, 1, 2, 1}}}},
     {{{{1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}}, {{1, 1, 1, 1, 2}, {1, 1, 1, 1, 2}, {2, 1, 1, 2, 1}}}}}};

INSTANTIATE_TEST_SUITE_P(
    smoke_AdaPool_1ch_Avg3DLayoutTest,
    AdaPoolLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::Combine(::testing::ValuesIn(std::vector<std::vector<int>>{{1},
                                                                                                               {2}}),
                                                             ::testing::ValuesIn(input3DShape1Channel)),
                                          ::testing::Values("avg"),
                                          ::testing::Values(true),
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::Values(CPUSpecificParams{{ncw, x}, {ncw}, {}, {}})),
    AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_AdaPool_1ch_Avg4DLayoutTest,
    AdaPoolLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::Combine(::testing::ValuesIn(std::vector<std::vector<int>>{{1, 1},
                                                                                                               {2, 2}}),
                                                             ::testing::ValuesIn(input4DShape1Channel)),
                                          ::testing::Values("avg"),
                                          ::testing::Values(true),
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::Values(CPUSpecificParams{{nchw, x}, {nchw}, {}, {}})),
    AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_AdaPool_1ch_Avg5DLayoutTest,
    AdaPoolLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::ValuesIn(std::vector<std::vector<int>>{{1, 1, 1}, {2, 2, 2}}),
                                              ::testing::ValuesIn(input5DShape1Channel)),
                           ::testing::Values("avg"),
                           ::testing::Values(true),
                           ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ::testing::Values(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}})),
    AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_AdaPool_1ch_Max3DLayoutTest,
    AdaPoolLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::Combine(::testing::ValuesIn(std::vector<std::vector<int>>{{1},
                                                                                                               {2}}),
                                                             ::testing::ValuesIn(input3DShape1Channel)),
                                          ::testing::Values("max"),
                                          ::testing::Values(true),
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::Values(CPUSpecificParams{{ncw, x}, {ncw}, {}, {}})),
    AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_AdaPool_1ch_Max4DLayoutTest,
    AdaPoolLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::Combine(::testing::ValuesIn(std::vector<std::vector<int>>{{1, 1},
                                                                                                               {2, 2}}),
                                                             ::testing::ValuesIn(input4DShape1Channel)),
                                          ::testing::Values("max"),
                                          ::testing::Values(true),
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::Values(CPUSpecificParams{{nchw, x}, {nchw}, {}, {}})),
    AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_AdaPool_1ch_Max5DLayoutTest,
    AdaPoolLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::ValuesIn(std::vector<std::vector<int>>{{1, 1, 1}, {2, 2, 2}}),
                                              ::testing::ValuesIn(input5DShape1Channel)),
                           ::testing::Values("max"),
                           ::testing::Values(true),
                           ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ::testing::Values(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}})),
    AdaPoolLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
