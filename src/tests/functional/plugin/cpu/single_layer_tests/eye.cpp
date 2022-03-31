/// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include <ngraph/opsets/opset9.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace {
    std::vector<int> pooledSpatialShape;
    std::string mode;
    std::vector<InputShape> inputShape;
}  // namespace

using AdaPoolSpecificParams = std::tuple<
        std::vector<int>,        // pooled vector
        std::vector<InputShape>>;      // feature map shape

using AdaPoolLayerTestParams = std::tuple<
        AdaPoolSpecificParams,
        std::string,         // mode
        bool,                // second Input is Constant
        ElementType,         // Net precision
        TargetDevice>;       // Device name

using EyeLikeLayerCPUTestParamsSet = std::tuple<
        CPULayerTestsDefinitions::AdaPoolLayerTestParams,
        CPUSpecificParams>;

class EyeLikeLayerCPUTest : public testing::WithParamInterface<EyeLikeLayerCPUTestParamsSet>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EyeLikeLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::AdaPoolLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType netPr;
        bool isStatic;
        AdaPoolSpecificParams adaPar;
        std::tie(adaPar, mode, isStatic, netPr, td) = basicParamsSet;
        std::tie(pooledSpatialShape, inputShape) = adaPar;
        std::ostringstream result;
        result << "AdaPoolTest_";
        result << "IS=(";
        for (const auto& shape : inputShape) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShape) {
            for (const auto& item : shape.second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
        }
        result << "OS=" << CommonTestUtils::vec2str(pooledSpatialShape) << "(spat.)_";
        result << netPr << "_";
        result << mode << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams) << "_";
        result << std::to_string(obj.index);
        return result.str();
    }
protected:
    void SetUp() override {
        CPULayerTestsDefinitions::AdaPoolLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        CPULayerTestsDefinitions::AdaPoolSpecificParams adaPoolParams;
        ElementType netPrecision;
        bool isStatic;
        std::tie(adaPoolParams, mode, isStatic, netPrecision, targetDevice) = basicParamsSet;
        std::tie(pooledVector, inputShape) = adaPoolParams;

        init_input_shapes(inputShape);
        if (!isStatic) {
            for (auto &target : targetStaticShapes) {
                target.push_back({pooledVector.size()});
            }
        }

        selectedType = std::string("unknown_FP32");
        if (netPrecision == ElementType::bf16) {
            rel_threshold = 1e-2;
        }
        function = createFunction(isStatic);
    }

    std::shared_ptr<ngraph::Function> createFunction(bool secondInputConst) {
        // auto params = ngraph::builder::makeDynamicParams(ngraph::element::f32, { inputDynamicShapes[0] });
        // params.front()->set_friendly_name("ParamsInput");
        // std::shared_ptr<ov::Node> secondInput;
        // if (secondInputConst) {
        //     secondInput = ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{pooledVector.size()}, pooledVector);
        // } else {
        //     auto pooledParam = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{pooledVector.size()});
        //     pooledParam->set_friendly_name("ParamSecondInput");
        //     params.push_back(pooledParam);
        //     secondInput = pooledParam;
        // }

        // auto adapoolMax = std::make_shared<ngraph::opset8::AdaptiveMaxPool>(params[0], secondInput, ngraph::element::i32);
        // adapoolMax->get_rt_info() = getCPUInfo();
        // auto adapoolAvg = std::make_shared<ngraph::opset8::AdaptiveAvgPool>(params[0], secondInput);
        // adapoolAvg->get_rt_info() = getCPUInfo();

        // auto function = (mode == "max" ? std::make_shared<ngraph::Function>(adapoolMax->outputs(), params, "AdaPoolMax") :
        //             std::make_shared<ngraph::Function>(adapoolAvg->outputs(), params, "AdaPoolAvg"));
        //
        auto rowsPar = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i32, ngraph::Shape{});
        rowsPar->set_friendly_name("rows");
        auto colsPar = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i32, ngraph::Shape{});
        colsPar->set_friendly_name("cols");
        auto eyelike = std::make_shared<ngraph::op::v9::Eye>(rowsPar, colsPar, ngraph::element::i32);
        eyelike->get_rt_info() = getCPUInfo();

        auto function2 = std::make_shared<ngraph::Function>(eyelike->outputs(), ngraph::ParameterVector{rowsPar, colsPar}, "EyeLike");

        //
        return function2;
    }

    //    void SetUp() override {
//        CPULayerTestsDefinitions::RangeLayerTestParams basicParamsSet;
//        CPUSpecificParams cpuParams;
//        std::tie(basicParamsSet, cpuParams) = this->GetParam();
//        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
//        CPULayerTestsDefinitions::RangeSpecificParams rangeParams;
//        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> shapes;
//        std::tie(rangeParams, inPrc, targetDevice) = basicParamsSet;
//        std::tuple<float, float, float> rangeInputs;
//
//        std::tie(shapes, rangeInputs, outPrc) = rangeParams;
//        targetStaticShapes = shapes.second;
//        inputDynamicShapes = shapes.first;
//
//        start = std::get<0>(rangeInputs);
//        stop = std::get<1>(rangeInputs);
//        step = std::get<2>(rangeInputs);
//        auto ngOutPr = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);
//        auto ngNetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
//        auto startPar = std::make_shared<ngraph::opset5::Parameter>(ngNetPrc, ngraph::Shape{});
//        auto stopPar = std::make_shared<ngraph::opset5::Parameter>(ngNetPrc, ngraph::Shape{});
//        auto stepPar = std::make_shared<ngraph::opset5::Parameter>(ngNetPrc, ngraph::Shape{});
//        auto range = std::make_shared<ngraph::opset4::Range>(startPar, stopPar, stepPar, ngOutPr);
//        range->get_rt_info() = getCPUInfo();
//        selectedType = std::string("ref_any_") + (inPrc == outPrc ? inPrc.name() : "FP32");
//        startPar->set_friendly_name("start");
//        stopPar->set_friendly_name("stop");
//        stepPar->set_friendly_name("step");
//
//        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(range)};
//        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector {
//            startPar, stopPar, stepPar}, "Range");
//        functionRefs = ngraph::clone_function(*function);
//    }
//};

    void init_ref_function(std::shared_ptr<ov::Model> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes) override {
        if (function->get_parameters().size() == 2) {
            funcRef = createFunction(true);
        }
        ngraph::helpers::resize_function(funcRef, targetInputStaticShapes);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (false && i == 1) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                auto *dataPtr = tensor.data<int32_t>();
                dataPtr[i] = 3;
            } else {
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 1, 3);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

private:
    std::vector<int> pooledVector;
};

TEST_P(EyeLikeLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    CheckPluginRelatedResults(compiledModel, "EyeLike");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice(std::string dims = "3D", std::string modeStr = "max") {
        std::vector<CPUSpecificParams> resCPUParams;
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {}, {}});
        return resCPUParams;
//     if (modeStr == "max") {
//         if (dims == "5D") {
//             resCPUParams.push_back(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}});  // i.e. two equal output layouts
//             resCPUParams.push_back(CPUSpecificParams{{ndhwc, x}, {ndhwc, ncdhw}, {}, {}});
//             if (with_cpu_x86_avx512f()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nCdhw16c, x}, {nCdhw16c, ncdhw}, {}, {}});
//             } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x}, {nCdhw8c, ncdhw}, {}, {}});
//             }
//         } else if (dims == "4D") {
//             resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw}, {}, {}});  // i.e. two equal output layouts
//             resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nchw}, {}, {}});
//             if (with_cpu_x86_avx512f()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nChw16c, x}, {nChw16c, nchw}, {}, {}});
//             } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c, nchw}, {}, {}});
//             }
//         } else {
//             resCPUParams.push_back(CPUSpecificParams{{ncw, x}, {ncw}, {}, {}});  // i.e. two equal output layouts
//             resCPUParams.push_back(CPUSpecificParams{{nwc, x}, {nwc, ncw}, {}, {}});
//             if (with_cpu_x86_avx512f()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nCw16c, x}, {nCw16c, ncw}, {}, {}});
//             } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nCw8c, x}, {nCw8c, ncw}, {}, {}});
//             }
//         }
//     } else {
//         if (dims == "5D") {
//             resCPUParams.push_back(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}});
//             resCPUParams.push_back(CPUSpecificParams{{ndhwc, x}, {ndhwc}, {}, {}});
//             if (with_cpu_x86_avx512f()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nCdhw16c, x}, {nCdhw16c}, {}, {}});
//             } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x}, {nCdhw8c}, {}, {}});
//             }
//         } else if (dims == "4D") {
//             resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw}, {}, {}});
//             resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc}, {}, {}});
//             if (with_cpu_x86_avx512f()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nChw16c, x}, {nChw16c}, {}, {}});
//             } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c}, {}, {}});
//             }
//         } else {
//             resCPUParams.push_back(CPUSpecificParams{{ncw, x}, {ncw}, {}, {}});
//             resCPUParams.push_back(CPUSpecificParams{{nwc, x}, {nwc}, {}, {}});
//             if (with_cpu_x86_avx512f()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nCw16c, x}, {nCw16c}, {}, {}});
//             } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
//                 resCPUParams.push_back(CPUSpecificParams{{nCw8c, x}, {nCw8c}, {}, {}});
//             }
//         }
//     }
//     return resCPUParams;
}

const std::vector<ElementType> netPrecisions = {
        ElementType::i32
};

const std::vector<std::vector<int>> pooled3DVector = {
        { 1 },
        { 3 },
        { 5 }
};
const std::vector<std::vector<int>> pooled4DVector = {
        { 1, 1 },
        { 3, 5 },
        { 5, 5 }
};

const std::vector<std::vector<int>> pooled5DVector = {
        { 1, 1, 1 },
        { 3, 5, 1 },
        { 3, 5, 3 },
};

std::vector<std::vector<ov::Shape>> staticInput3DShapeVector = {{{}, {}}};

const std::vector<std::vector<InputShape>> input3DShapeVector = {
        {
                {{{-1, 17, -1}, {{1, 17, 3}, {3, 17, 5}, {3, 17, 5}}}},
                {{{{1, 10}, 20, {1, 10}}, {{1, 20, 5}, {2, 20, 4}, {3, 20, 6}}}}
        }
};

std::vector<std::vector<ov::Shape>> staticInput4DShapeVector = {{{1, 3, 1, 1}, {3, 17, 5, 2}}};

const std::vector<std::vector<InputShape>> input4DShapeVector = {
        {
                {{{-1, 3, -1, -1}, {{1, 3, 1, 1}, {3, 3, 5, 2}, {3, 3, 5, 2}}}},
                {{{{1, 10}, 3, {1, 10}, {1, 10}}, {{2, 3, 10, 6}, {3, 3, 6, 5}, {3, 3, 6, 5}}}}
        }
};

std::vector<std::vector<ov::Shape>> staticInput5DShapeVector = {{{ 1, 17, 2, 5, 2}, {3, 17, 4, 5, 4}}};

const std::vector<std::vector<InputShape>> input5DShapeVector = {
        {
                {{{-1, 17, -1, -1, -1}, {{1, 17, 2, 5, 2}, {3, 17, 4, 5, 4}, {3, 17, 4, 5, 4}}}},
                {{{{1, 10}, 3, {1, 10}, {1, 10}, {1, 10}}, {{3, 3, 2, 5, 2}, {1, 3, 4, 5, 4}, {1, 3, 4, 5, 4}}}}
        }
};

const auto adaPool3DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled3DVector),         // output spatial shape
        ::testing::ValuesIn(input3DShapeVector)      // feature map shape
);

const auto adaPool4DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled4DVector),         // output spatial shape
        ::testing::ValuesIn(input4DShapeVector)     // feature map shape
);

const auto adaPool5DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled5DVector),         // output spatial shape
        ::testing::ValuesIn(input5DShapeVector)     // feature map shape
);

std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapesDynamic = {
       {{ngraph::PartialShape(), ngraph::PartialShape()},
        {{ngraph::Shape{}, ngraph::Shape{}}, {ngraph::Shape{}, ngraph::Shape{}}}}
};

const auto staticAdaPool3DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled3DVector),         // output spatial shape
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInput3DShapeVector))      // feature map shape
);

const auto staticAdaPool4DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled4DVector),         // output spatial shape
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInput4DShapeVector))     // feature map shape
);

const auto staticAdaPool5DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled5DVector),         // output spatial shape
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInput5DShapeVector))     // feature map shape
);

INSTANTIATE_TEST_SUITE_P(x_smoke_StaticAdaPoolAvg3DLayoutTest, EyeLikeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         staticAdaPool3DParams,
                                         ::testing::Values("avg"),
                                         ::testing::Values(true),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("3D", "avg"))),
                         EyeLikeLayerCPUTest::getTestCaseName);

// in 1-channel cases  {..., 1, 1, 1} shape cannot be correctly resolved on oneDnn level, so it was removed from instances

const std::vector<std::vector<InputShape>> input3DShape1Channel = {
        {
                {{{-1, -1, -1}, {{1, 1, 2}, {1, 1, 2}, {1, 1, 2}}}},
                {{{{1, 10}, {1, 10}, {1, 10}}, {{1, 1, 2}, {2, 1, 2}, {2, 1, 2}}}}
        }
};

const std::vector<std::vector<InputShape>> input4DShape1Channel = {
        {
                {{{-1, -1, -1, -1}, {{1, 1, 1, 2}, {2, 1, 2, 1}, {2, 1, 2, 1}}}},
                {{{{1, 10}, {1, 10}, {1, 10}, {1, 10}}, {{1, 1, 1, 2}, {1, 1, 1, 2}, {2, 1, 2, 1}}}}
        }
};

const std::vector<std::vector<InputShape>> input5DShape1Channel = {
        {
                {{{-1, -1, -1, -1, -1}, {{1, 1, 1, 1, 2}, {1, 1, 1, 1, 2}, {2, 1, 1, 2, 1}}}},
                {{{{1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}}, {{1, 1, 1, 1, 2}, {1, 1, 1, 1, 2}, {2, 1, 1, 2, 1}}}}
        }
};

// INSTANTIATE_TEST_SUITE_P(x_smoke_AdaPool_1ch_Avg3DLayoutTest, EyeLikeLayerCPUTest,
//                          ::testing::Combine(
//                                  ::testing::Combine(
//                                          ::testing::Combine(
//                                                  ::testing::ValuesIn(std::vector<std::vector<int>> {
//                                                          {1}, {2}}),
//                                                  ::testing::ValuesIn(input3DShape1Channel)),
//                                          ::testing::Values("avg"),
//                                          ::testing::Values(true),
//                                          ::testing::ValuesIn(netPrecisions),
//                                          ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                                  ::testing::Values(CPUSpecificParams{{}, {}, {}, {}})),
//                          EyeLikeLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions