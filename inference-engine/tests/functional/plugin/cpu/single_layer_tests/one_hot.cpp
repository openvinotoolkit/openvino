// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph_functions/builders.hpp>
#include <functional_test_utils/ov_tensor_utils.hpp>
#include <random>
#include <ctime>
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using OneHotInputShapes = std::pair<std::vector<ov::PartialShape>, std::vector<ov::Shape>>;

using oneHotCPUTestParams = std::tuple<
        OneHotInputShapes,                 // Input shape
        int,                               // axis to extend
        ngraph::helpers::InputLayerType,   // secondary input type
        size_t,                            // depth
        float,                             // on_value
        float,                             // off_value
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        CPUSpecificParams>;

class OneHotLayerCPUTest : public testing::WithParamInterface<oneHotCPUTestParams>,
                           virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<oneHotCPUTestParams>& obj) {
        OneHotInputShapes inputShape;
        int axis;
        ngraph::helpers::InputLayerType inputType;
        size_t depth;
        float onValue, offValue;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        CPUSpecificParams cpuParams;
        std::tie(inputShape, axis, inputType, depth, onValue, offValue, netPrecision, inPrc, outPrc, cpuParams) = obj.param;

        std::ostringstream result;
        if (!inputShape.first.empty()) {
            result << "IS=(";
            result << CommonTestUtils::partialShape2str(inputShape.first) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << "axis=" << axis << "_";
        if (inputType == ngraph::helpers::InputLayerType::CONSTANT)
            result << "depth=" << depth << "_";
        else
            result << "depth=PARAMETER" << "_";
        result << "OnVal=" << onValue << "_";
        result << "OffVal=" << offValue << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name();
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;

            if (i == 1) {
                tensor = ov::runtime::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
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
        targetDevice = CommonTestUtils::DEVICE_CPU;

        OneHotInputShapes inputShape;
        ngraph::helpers::InputLayerType inputType;
        InferenceEngine::Precision netPrecision, inPrc, outPrc;
        CPUSpecificParams cpuParams;
        std::tie(inputShape, Axis, inputType, Depth, OnValue, OffValue, netPrecision, inPrc, outPrc, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType = std::string("ref_any_") + inPrc.name();
        inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        outType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);

        if (!inputShape.first.empty()) {
            inputDynamicShapes = inputShape.first;
        } else {
            inputDynamicShapes.emplace_back(inputShape.second.front());
        }
        for (const auto& target : inputShape.second) {
            targetStaticShapes.push_back({target});
        }
        if (inputType == ngraph::helpers::InputLayerType::PARAMETER) {
            for (auto &target : targetStaticShapes)
                target.push_back({});
            function = createFunctionWithParam();
        } else {
            function = createFunctionWithConst();
        }
    }
    std::shared_ptr<ngraph::Function> createFunctionWithConst() {
        auto params = ngraph::builder::makeDynamicParams(ngraph::element::i32, {inputDynamicShapes});
        for (auto& p : params)
            p->set_friendly_name("ParamsIndices");
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));
        auto depth_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i32, ngraph::Shape{ }, Depth);
        auto on_value_const = std::make_shared<ngraph::op::Constant>(outType, ngraph::Shape{ }, OnValue);
        auto off_value_const = std::make_shared<ngraph::op::Constant>(outType, ngraph::Shape{ }, OffValue);
        auto oneHot = std::make_shared<ngraph::opset5::OneHot>(paramOuts[0], depth_const, on_value_const, off_value_const, Axis);
        return makeNgraphFunction(ngraph::element::i32, params, oneHot, "OneHot");
    }
    std::shared_ptr<ngraph::Function> createFunctionWithParam() {
        auto params = ngraph::builder::makeDynamicParams(ngraph::element::i32, {inputDynamicShapes});
        for (auto& p : params)
            p->set_friendly_name("ParamsIndices");
        auto depth_param = std::make_shared<ngraph::op::Parameter>(ngraph::element::i32, ngraph::Shape{ });
        depth_param->set_friendly_name("ParamDepth");
        params.push_back(depth_param);
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));
        auto on_value_const = std::make_shared<ngraph::op::Constant>(outType, ngraph::Shape{ }, OnValue);
        auto off_value_const = std::make_shared<ngraph::op::Constant>(outType, ngraph::Shape{ }, OffValue);
        auto oneHot = std::make_shared<ngraph::opset5::OneHot>(paramOuts[0], paramOuts[1], on_value_const, off_value_const, Axis);
        return makeNgraphFunction(ngraph::element::i32, params, oneHot, "OneHot");
    }
    void generateDepth() {
        std::mt19937 gen;
        gen.seed(static_cast<unsigned int>(time(0)));
        Depth = gen() % 9 + 1;
    }

    int Axis;
    size_t Depth;
    float OnValue, OffValue;
};

TEST_P(OneHotLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    if (function->get_parameters().size() == 1) {
        run();
    } else {
        compile_model();
        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            try {
                generateDepth();
                if (!inputDynamicShapes.empty()) {
                    // resize ngraph function according new target shape
                    functionRefs = createFunctionWithConst();
                    auto inputs = functionRefs->inputs();
                    std::map<ov::Output<ov::Node>, ov::PartialShape> shapes;
                    if (inputs.size() > targetStaticShapeVec.size()) {
                        throw std::runtime_error("targetInputStaticShapes.size() = " + std::to_string(targetStaticShapeVec.size()) +
                                                 " != inputs.size() = " + std::to_string(inputs.size()));
                    }
                    for (size_t i = 0; i < inputs.size(); i++) {
                        shapes.insert({inputs[i], targetStaticShapeVec[i]});
                    }
                    functionRefs->reshape(shapes);
                }
                generate_inputs(targetStaticShapeVec);
                infer();
                for (const auto& in : inputs) {
                    if (strcmp(in.first->get_friendly_name().data(), "ParamDepth") == 0) {
                        inputs.erase(in.first);
                        break;
                    }
                }
                validate();
            } catch (const std::exception &ex) {
                throw std::runtime_error("Incorrect target static shape: " + CommonTestUtils::vec2str(targetStaticShapeVec) + " " + ex.what());
            }
        }
    }
    // TODO: Should be uncommented after updating the CheckPluginRelatedResults() method
    // CheckPluginRelatedResults(executableNetwork, "OneHot");
}

namespace {
const std::vector<Precision> outPrc = {
        Precision::FP32,
        // TODO: Should be uncommented after PR #8339 merge
        // Precision::BF16,
        Precision::I8,
        Precision::U8
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER
};

const std::vector<OneHotInputShapes> staticInputShapes0D = {
        { {}, {{}} }
};

// 0d -> 1d, depth
const auto testCase_1d = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes0D),
        ::testing::Values(-1, 0),
        ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
        ::testing::Values(3),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::Values(Precision::I32),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_1D, OneHotLayerCPUTest, testCase_1d, OneHotLayerCPUTest::getTestCaseName);

const std::vector<OneHotInputShapes> staticInputShapes1D = {
        { {}, {{3}} }
};
// 1d -> 2d, axis default
const auto testCase_2d_static = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes1D),
        ::testing::Values(-1, 0, 1),
        ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
        ::testing::Values(6),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::Values(Precision::I32),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_2D_Static, OneHotLayerCPUTest, testCase_2d_static, OneHotLayerCPUTest::getTestCaseName);

const std::vector<OneHotInputShapes> dynamicInputShapes1D = {
        {
                // dynamic
                {{-1}},
                // target
                {
                        {3},
                        {4},
                        {5}
                }
        },
        {
                // dynamic
                {{{1, 5}}},
                // target
                {
                        {1},
                        {3},
                        {5}
                }
        }
};
// 1d -> 2d, axis default
const auto testCase_2d_dynamic = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes1D),
        ::testing::Values(-1, 0, 1),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::Values(6),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::Values(Precision::I32),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_2D_Dynamic, OneHotLayerCPUTest, testCase_2d_dynamic, OneHotLayerCPUTest::getTestCaseName);

const std::vector<OneHotInputShapes> staticInputShapes2D = {
        { {}, {{3, 2}} }
};
// 2d -> 3d, on_value, off_value
const auto testCase_3d_static = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes2D),
        ::testing::Values(-1, 0, 1),
        ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
        ::testing::Values(4),
        ::testing::Values(2.f),
        ::testing::Values(-1.f),
        ::testing::Values(Precision::I32),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_3D_Static, OneHotLayerCPUTest, testCase_3d_static, OneHotLayerCPUTest::getTestCaseName);

const std::vector<OneHotInputShapes> dynamicInputShapes2D = {
        {
                // dynamic
                {{-1, -1}},
                // target
                {
                        {3, 2},
                        {2, 3},
                        {4, 4}
                }
        },
        {
                // dynamic
                {{-1, 3}},
                // target
                {
                        {2, 3},
                        {3, 3},
                        {4, 3}
                }
        },
        {
                // dynamic
                {{{1, 5}, {3, 4}}},
                // target
                {
                        {2, 3},
                        {3, 4},
                        {4, 3}
                }
        }
};
// 2d -> 3d, on_value, off_value
const auto testCase_3d_dynamic = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes2D),
        ::testing::Values(-1, 0, 1),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::Values(4),
        ::testing::Values(2.f),
        ::testing::Values(-1.f),
        ::testing::Values(Precision::I32),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_3D_Dynamic, OneHotLayerCPUTest, testCase_3d_dynamic, OneHotLayerCPUTest::getTestCaseName);

const std::vector<OneHotInputShapes> staticInputShapes3D = {
        { {}, {{1, 3, 2}} }
};
// 3d -> 4d
const auto testCase_4d_static = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes3D),
        ::testing::Values(-1, 0, 1, 2),
        ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
        ::testing::Values(4),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::Values(Precision::I32),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_4D_Static, OneHotLayerCPUTest, testCase_4d_static, OneHotLayerCPUTest::getTestCaseName);

const std::vector<OneHotInputShapes> dynamicInputShapes3D = {
        {
                // dynamic
                {{-1, -1, -1}},
                // target
                {
                        {1, 3, 2},
                        {1, 2, 3},
                        {2, 4, 4}
                }
        },
        {
                // dynamic
                {{-1, 3, -1}},
                // target
                {
                        {2, 3, 1},
                        {1, 3, 2},
                        {1, 3, 5}
                }
        },
        {
                // dynamic
                {{{1, 2}, 3, {1, 5}}},
                // target
                {
                        {2, 3, 1},
                        {1, 3, 2},
                        {1, 3, 5}
                }
        }
};
// 3d -> 4d
const auto testCase_4d_dynamic = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes3D),
        ::testing::Values(-1, 0, 1, 2),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::Values(4),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::Values(Precision::I32),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_4D_Dynamic, OneHotLayerCPUTest, testCase_4d_dynamic, OneHotLayerCPUTest::getTestCaseName);

const std::vector<OneHotInputShapes> staticInputShapes4D = {
        { {}, {{1, 3, 2, 3}} }
};
// 4d -> 5d
const auto testCase_5d_static = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes4D),
        ::testing::Values(-1, 0, 1, 2, 3),
        ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
        ::testing::Values(4),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::Values(Precision::I32),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_5D_Static, OneHotLayerCPUTest, testCase_5d_static, OneHotLayerCPUTest::getTestCaseName);

const std::vector<OneHotInputShapes> dynamicInputShapes4D = {
        {
                // dynamic
                {{-1, -1, -1, -1}},
                // target
                {
                        {1, 3, 2, 3},
                        {1, 2, 3, 2},
                        {2, 3, 4, 4}
                }
        },
        {
                // dynamic
                {{-1, 3, -1, {1, 3}}},
                // target
                {
                        {1, 3, 3, 1},
                        {1, 3, 2, 2},
                        {1, 3, 5, 3}
                }
        },
        {
                // dynamic
                {{{1, 2}, 3, {2, 5}, {1, 3}}},
                // target
                {
                        {1, 3, 3, 1},
                        {2, 3, 2, 2},
                        {1, 3, 5, 3}
                }
        }
};
// 4d -> 5d
const auto testCase_5d_dynamic = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D),
        ::testing::Values(-1, 0, 1, 2, 3),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::Values(4),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::Values(Precision::I32),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_5D_Dynamic, OneHotLayerCPUTest, testCase_5d_dynamic, OneHotLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
