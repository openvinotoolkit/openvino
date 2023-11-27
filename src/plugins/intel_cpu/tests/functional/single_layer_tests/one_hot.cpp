// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_models/builders.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using oneHotCPUTestParams = std::tuple<
        InputShape,                                        // Input shape
        int,                                               // axis to extend
        std::pair<ngraph::helpers::InputLayerType, bool>,  // secondary input type && need to generate depth
        size_t,                                            // depth
        float,                                             // on_value
        float,                                             // off_value
        InferenceEngine::Precision,                        // Output precision
        CPUSpecificParams>;

class OneHotLayerCPUTest : public testing::WithParamInterface<oneHotCPUTestParams>,
                           virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<oneHotCPUTestParams>& obj) {
        InputShape inputShape;
        int axis;
        std::pair<ngraph::helpers::InputLayerType, bool> inputType;
        size_t depth;
        float onValue, offValue;
        InferenceEngine::Precision outPrc;
        CPUSpecificParams cpuParams;
        std::tie(inputShape, axis, inputType, depth, onValue, offValue, outPrc, cpuParams) = obj.param;

        std::ostringstream result;
        if (inputShape.first.size() != 0) {
            result << "IS=(" << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShape.second) {
                result << ov::test::utils::vec2str(shape) << "_";
        }
        result << "axis=" << axis << "_";
        if (inputType.first == ngraph::helpers::InputLayerType::CONSTANT && !inputType.second) {
            result << "depth=" << depth << "_";
        } else if (inputType.first == ngraph::helpers::InputLayerType::CONSTANT && inputType.second) {
            result << "depth=WillBeGenerated" << "_";
        } else {
            result << "depth=PARAMETER" << "_";
        }
        result << "OnVal=" << onValue << "_";
        result << "OffVal=" << offValue << "_";
        result << "outPRC=" << outPrc.name();
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 1) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                auto *dataPtr = tensor.data<int32_t>();
                dataPtr[0] = Depth;
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        InputShape inputShape;
        std::pair<ngraph::helpers::InputLayerType, bool> inputType;
        InferenceEngine::Precision outPrc;
        CPUSpecificParams cpuParams;
        std::tie(inputShape, Axis, inputType, Depth, OnValue, OffValue, outPrc, cpuParams) = this->GetParam();

        if (inputType.second && inputType.first == ngraph::helpers::InputLayerType::CONSTANT) {
            generateDepth();
        }

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType = std::string("ref_any_I32");
        outType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);

        init_input_shapes({inputShape});
        if (inputType.second) {
            for (auto &target : targetStaticShapes)
                target.push_back({});
        }

        function = createFunction(inputType.first == ngraph::helpers::InputLayerType::CONSTANT);
        if (function->get_parameters().size() == 2) {
            generateDepth();
            functionRefs = createFunction(true);
        }
    }
    void validate() override {
            auto actualOutputs = get_plugin_outputs();
        if (function->get_parameters().size() == 2) {
            auto pos = std::find_if(inputs.begin(), inputs.end(),
                                    [](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor> &params) {
                                        return params.first->get_friendly_name() == "ParamDepth";
                                    });
            OPENVINO_ASSERT(pos != inputs.end());
            inputs.erase(pos);
        }
        auto expectedOutputs = calculate_refs();
        if (expectedOutputs.empty()) {
                return;
        }
        ASSERT_EQ(actualOutputs.size(), expectedOutputs.size())
                << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

        compare(expectedOutputs, actualOutputs);
    }
    std::shared_ptr<ngraph::Function> createFunction(bool depthConst) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngraph::element::i32, inputDynamicShapes.front())};
        params.front()->set_friendly_name("ParamsIndices");
        std::shared_ptr<ov::Node> depth;
        if (depthConst) {
            depth = ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{ }, {Depth});
        } else {
            auto depthParam = std::make_shared<ngraph::op::Parameter>(ngraph::element::i32, ngraph::Shape{ });
            depthParam->set_friendly_name("ParamDepth");
            params.push_back(depthParam);
            depth = depthParam;
        }
        auto on_value_const = std::make_shared<ngraph::op::Constant>(outType, ngraph::Shape{ }, OnValue);
        auto off_value_const = std::make_shared<ngraph::op::Constant>(outType, ngraph::Shape{ }, OffValue);
        auto oneHot = std::make_shared<ngraph::opset5::OneHot>(params[0], depth, on_value_const, off_value_const, Axis);
        return makeNgraphFunction(ngraph::element::i32, params, oneHot, "OneHot");
    }
    void generateDepth() {
        testing::internal::Random random(time(nullptr));
        random.Generate(10);
        Depth = static_cast<int64_t>(1 + static_cast<int64_t>(random.Generate(10)));
    }

    int Axis;
    size_t Depth;
    float OnValue, OffValue;
};

TEST_P(OneHotLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "OneHot");
}

namespace {
const std::vector<Precision> outPrc = {
        Precision::FP32,
        Precision::BF16,
        Precision::I8
        // Precision::U8  // Precision cannot be wrapped to constant one hot
};

std::vector<std::pair<ngraph::helpers::InputLayerType, bool>> secondaryInputTypesStaticCase = {
        {ngraph::helpers::InputLayerType::CONSTANT, true},
        {ngraph::helpers::InputLayerType::CONSTANT, false}
};
std::vector<std::pair<ngraph::helpers::InputLayerType, bool>> secondaryInputTypesDynamicCase = {
        {ngraph::helpers::InputLayerType::CONSTANT, true},
        {ngraph::helpers::InputLayerType::CONSTANT, false},
        {ngraph::helpers::InputLayerType::PARAMETER, true}
};

const std::vector<ov::Shape> staticInputShapes0D = {
        { }
};

// 0d -> 1d, depth
const auto testCase_1d = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes0D)),
        ::testing::Values(-1, 0),
        ::testing::ValuesIn(secondaryInputTypesStaticCase),
        ::testing::Values(3),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_1D, OneHotLayerCPUTest, testCase_1d, OneHotLayerCPUTest::getTestCaseName);

const std::vector<ov::Shape> staticInputShapes1D = {
        { 3 }
};
// 1d -> 2d, axis default
const auto testCase_2d_static = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes1D)),
        ::testing::Values(-1, 0, 1),
        ::testing::ValuesIn(secondaryInputTypesStaticCase),
        ::testing::Values(6),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_2D_Static, OneHotLayerCPUTest, testCase_2d_static, OneHotLayerCPUTest::getTestCaseName);

const std::vector<InputShape> dynamicInputShapes1D = {
        {{-1}, {{3}, {4}, {5}}},
        {{{1, 5}}, {{1}, {3}, {5}}},
};
// 1d -> 2d, axis default
const auto testCase_2d_dynamic = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes1D),
        ::testing::Values(-1, 0, 1),
        ::testing::ValuesIn(secondaryInputTypesDynamicCase),
        ::testing::Values(6),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_2D_Dynamic, OneHotLayerCPUTest, testCase_2d_dynamic, OneHotLayerCPUTest::getTestCaseName);

const std::vector<ov::Shape> staticInputShapes2D = {
        { 3, 2 }
};
// 2d -> 3d, on_value, off_value
const auto testCase_3d_static = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes2D)),
        ::testing::Values(-1, 0, 1),
        ::testing::ValuesIn(secondaryInputTypesStaticCase),
        ::testing::Values(4),
        ::testing::Values(2.f),
        ::testing::Values(-1.f),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_3D_Static, OneHotLayerCPUTest, testCase_3d_static, OneHotLayerCPUTest::getTestCaseName);

const std::vector<InputShape> dynamicInputShapes2D = {
        {{-1, -1}, {{3, 2}, {2, 3}, {4, 4}}},
        {{-1, 3}, {{2, 3}, {3, 3}, {4, 3}}},
        {{{1, 5}, {3, 4}}, {{2, 3}, {3, 4}, {4, 3}}}
};
// 2d -> 3d, on_value, off_value
const auto testCase_3d_dynamic = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes2D),
        ::testing::Values(-1, 0, 1),
        ::testing::ValuesIn(secondaryInputTypesDynamicCase),
        ::testing::Values(4),
        ::testing::Values(2.f),
        ::testing::Values(-1.f),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_3D_Dynamic, OneHotLayerCPUTest, testCase_3d_dynamic, OneHotLayerCPUTest::getTestCaseName);

const std::vector<ov::Shape> staticInputShapes3D = {
        { 1, 3, 2 }
};
// 3d -> 4d
const auto testCase_4d_static = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes3D)),
        ::testing::Values(-1, 0, 1, 2),
        ::testing::ValuesIn(secondaryInputTypesStaticCase),
        ::testing::Values(4),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_4D_Static, OneHotLayerCPUTest, testCase_4d_static, OneHotLayerCPUTest::getTestCaseName);

const std::vector<InputShape> dynamicInputShapes3D = {
        {{-1, -1, -1}, {{1, 3, 2}, {1, 2, 3}, {2, 4, 4}}},
        {{-1, 3, -1}, {{2, 3, 1}, {1, 3, 2}, {1, 3, 5}}},
        {{{1, 2}, 3, {1, 5}}, {{2, 3, 1}, {1, 3, 2}, {1, 3, 5}}}
};
// 3d -> 4d
const auto testCase_4d_dynamic = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes3D),
        ::testing::Values(-1, 0, 1, 2),
        ::testing::ValuesIn(secondaryInputTypesDynamicCase),
        ::testing::Values(4),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_4D_Dynamic, OneHotLayerCPUTest, testCase_4d_dynamic, OneHotLayerCPUTest::getTestCaseName);

const std::vector<ov::Shape> staticInputShapes4D = {
        { 1, 3, 2, 3 }
};
// 4d -> 5d
const auto testCase_5d_static = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes4D)),
        ::testing::Values(-1, 0, 1, 2, 3),
        ::testing::ValuesIn(secondaryInputTypesStaticCase),
        ::testing::Values(4),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_5D_Static, OneHotLayerCPUTest, testCase_5d_static, OneHotLayerCPUTest::getTestCaseName);

const std::vector<InputShape> dynamicInputShapes4D = {
        {{-1, -1, -1, -1}, {{1, 3, 2, 3}, {1, 2, 3, 2}, {2, 3, 4, 4}}},
        {{-1, 3, -1, {1, 3}}, {{1, 3, 3, 1}, {1, 3, 2, 2}, {1, 3, 5, 3}}},
        {{{1, 2}, 3, {2, 5}, {1, 3}}, {{1, 3, 3, 1}, {2, 3, 2, 2}, {1, 3, 5, 3}}}
};
// 4d -> 5d
const auto testCase_5d_dynamic = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D),
        ::testing::Values(-1, 0, 1, 2, 3),
        ::testing::ValuesIn(secondaryInputTypesDynamicCase),
        ::testing::Values(4),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_5D_Dynamic, OneHotLayerCPUTest, testCase_5d_dynamic, OneHotLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
