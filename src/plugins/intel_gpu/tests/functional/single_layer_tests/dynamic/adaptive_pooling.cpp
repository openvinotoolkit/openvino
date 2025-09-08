// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <random>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/adaptive_avg_pool.hpp"
#include "openvino/op/adaptive_max_pool.hpp"

namespace {
using ov::test::InputShape;
using AdaPoolSpecificParams = std::tuple<std::vector<int>,          // pooled vector
                                         std::vector<InputShape>>;  // feature map shape

using AdaPoolLayerGPUTestParams = std::tuple<AdaPoolSpecificParams,
                                             std::string,          // mode
                                             bool,                 // second Input is Constant
                                             ov::element::Type,    // Net precision
                                             std::string>;         // Device name


class AdaPoolLayerGPUTest : public testing::WithParamInterface<AdaPoolLayerGPUTestParams>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AdaPoolLayerGPUTestParams> obj) {
        AdaPoolLayerGPUTestParams basicParamsSet;
        basicParamsSet = obj.param;

        const auto& [adaPar, mode, isStatic, netPr, targetDevice] = basicParamsSet;
        const auto& [pooledSpatialShape, inputShape] = adaPar;
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
        result << "device=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        AdaPoolLayerGPUTestParams basicParamsSet;
        basicParamsSet = this->GetParam();

        const auto& [adaPoolParams, _mode, isStatic, netPrecision, _targetDevice] = basicParamsSet;
        mode = _mode;
        targetDevice = _targetDevice;
        const auto& [_pooledVector, inputShape] = adaPoolParams;
        pooledVector = _pooledVector;

        init_input_shapes(inputShape);
        if (!isStatic) {
            for (auto& target : targetStaticShapes) {
                target.push_back({pooledVector.size()});
            }
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
            // ngraph shape infer for adaptive pooling has seg fault when i32 type of second input
            secondInput = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{pooledVector.size()}, pooledVector);
        } else {
            auto pooledParam =
                // ngraph shape infer for adaptive pooling has seg fault when i32 type of second input
                std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{pooledVector.size()});
            pooledParam->set_friendly_name("ParamSecondInput");
            params.push_back(pooledParam);
            secondInput = pooledParam;
        }

        auto adapoolMax = std::make_shared<ov::op::v8::AdaptiveMaxPool>(params[0], secondInput, ov::element::i64);
        auto adapoolAvg = std::make_shared<ov::op::v8::AdaptiveAvgPool>(params[0], secondInput);

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
                auto* dataPtr = tensor.data<int64_t>();
                for (size_t i = 0; i < pooledVector.size(); i++) {
                    // ngraph shape infer for adaptive pooling has seg fault when i32 type of second input
                    dataPtr[i] = static_cast<int64_t>(pooledVector[i]);
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

TEST_P(AdaPoolLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> netPrecisions = {ov::element::f32, ov::element::f16};

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

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolAvg3DLayoutTest, AdaPoolLayerGPUTest,
                        ::testing::Combine(
                            adaPool3DParams,
                            ::testing::Values("avg"),
                            ::testing::Values(false),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        AdaPoolLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolAvg4DLayoutTest, AdaPoolLayerGPUTest,
                        ::testing::Combine(
                            adaPool4DParams,
                            ::testing::Values("avg"),
                            ::testing::Values(false),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        AdaPoolLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolAvg5DLayoutTest, AdaPoolLayerGPUTest,
                        ::testing::Combine(
                            adaPool5DParams,
                            ::testing::Values("avg"),
                            ::testing::Values(false),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        AdaPoolLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolMax3DLayoutTest, AdaPoolLayerGPUTest,
                        ::testing::Combine(
                            adaPool3DParams,
                            ::testing::Values("max"),
                            ::testing::Values(false),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        AdaPoolLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolMax4DLayoutTest, AdaPoolLayerGPUTest,
                        ::testing::Combine(
                            adaPool4DParams,
                            ::testing::Values("max"),
                            ::testing::Values(false),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        AdaPoolLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolMax5DLayoutTest, AdaPoolLayerGPUTest,
                        ::testing::Combine(
                            adaPool5DParams,
                            ::testing::Values("max"),
                            ::testing::Values(false),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        AdaPoolLayerGPUTest::getTestCaseName);

const auto staticAdaPool3DParams = ::testing::Combine(
    ::testing::ValuesIn(pooled3DVector),                                                 // output spatial shape
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(staticInput3DShapeVector))  // feature map shape
);

const auto staticAdaPool4DParams = ::testing::Combine(
    ::testing::ValuesIn(pooled4DVector),                                                 // output spatial shape
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(staticInput4DShapeVector))  // feature map shape
);

const auto staticAdaPool5DParams = ::testing::Combine(
    ::testing::ValuesIn(pooled5DVector),                                                 // output spatial shape
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(staticInput5DShapeVector))  // feature map shape
);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_StaticAdaPoolAvg3DLayoutTest, AdaPoolLayerGPUTest,
                         ::testing::Combine(staticAdaPool3DParams,
                                            ::testing::Values("avg"),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         AdaPoolLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_StaticAdaPoolAvg4DLayoutTest, AdaPoolLayerGPUTest,
                         ::testing::Combine(staticAdaPool4DParams,
                                            ::testing::Values("avg"),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         AdaPoolLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_StaticAdaPoolAvg5DLayoutTest, AdaPoolLayerGPUTest,
                         ::testing::Combine(staticAdaPool5DParams,
                                            ::testing::Values("avg"),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         AdaPoolLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_StaticAdaPoolMax3DLayoutTest, AdaPoolLayerGPUTest,
                         ::testing::Combine(staticAdaPool3DParams,
                                            ::testing::Values("max"),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         AdaPoolLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_StaticAdaPoolMax4DLayoutTest, AdaPoolLayerGPUTest,
                         ::testing::Combine(staticAdaPool4DParams,
                                            ::testing::Values("max"),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         AdaPoolLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_StaticAdaPoolMax5DLayoutTest, AdaPoolLayerGPUTest,
                         ::testing::Combine(staticAdaPool5DParams,
                                            ::testing::Values("max"),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         AdaPoolLayerGPUTest::getTestCaseName);
} // namespace
