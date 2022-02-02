//// Copyright (C) 2018-2022 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#include <shared_test_classes/single_layer/gather.hpp>
//#include "ngraph_functions/builders.hpp"
//#include "test_utils/cpu_test_utils.hpp"
//
//using namespace InferenceEngine;
//using namespace CPUTestUtils;
//
//namespace CPULayerTestsDefinitions {
//
//using inputShapesPair = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;
//
//typedef std::tuple<
//        inputShapesPair,                   // Input shapes
//        int64_t,                           // Axis
//        int64_t,                           // Batch dims
//        InferenceEngine::Precision,        // Network precision
//        bool,                              // Is axis input constant
//        std::string,                       // Device name
//        CPUSpecificParams                  // CPU specific params
//> GatherLayerTestCPUParams;
//
//class GatherLayerTestCPU : public testing::WithParamInterface<GatherLayerTestCPUParams>,
//                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
//public:
//    static std::string getTestCaseName(testing::TestParamInfo<GatherLayerTestCPUParams> obj) {
//        inputShapesPair inputShapes;
//        int axis, batchDims;
//        Precision netPrecision;
//        std::string targetDevice;
//        bool isAxisConstant;
//        CPUSpecificParams cpuParams;
//        std::tie(inputShapes, axis, batchDims, netPrecision, isAxisConstant, targetDevice, cpuParams) = obj.param;
//
//        std::ostringstream result;
//        result << "DynShapes=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
//        result << "StatShapes=" << CommonTestUtils::vec2str(inputShapes.second) << "_";
//        result << "axis=" << axis << "_";
//        result << "batchDims=" << batchDims << "_";
//        result << "netPrc=" << netPrecision.name() << "_";
//        result << "constAx=" << (isAxisConstant ? "True" : "False") << "_";
//        result << "trgDev=" << targetDevice;
//        result << CPUTestsBase::getTestCaseName(cpuParams);
//
//        return result.str();
//    }
//
//protected:
//    void SetUp() override {
//        inputShapesPair inputShapes;
//        int64_t batchDims;
//        Precision netPrecision;
//        CPUSpecificParams cpuParams;
//        bool isAxisConstant = true;
//        std::tie(inputShapes, axis, batchDims, netPrecision, isAxisConstant, targetDevice, cpuParams) = this->GetParam();
//
//        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
//
//        selectedType = std::string("ref_any_") + netPrecision.name();
//
//        targetStaticShapes.reserve(inputShapes.second.size());
//        inputDynamicShapes.reserve(inputShapes.first.size());
//        for (int i = 0; i < (isAxisConstant ? 2 : 3); i++) {
//            if (inputShapes.second.size() > i)
//                targetStaticShapes.push_back({inputShapes.second[i]});
//            if (inputShapes.first.size() > i)
//                inputDynamicShapes.push_back(inputShapes.first[i]);
//        }
//        const ov::Shape& inputDataShape = targetStaticShapes.front().front(), indicesShape = targetStaticShapes.front()[1];
//        dataSrcRank = inputDataShape.size();
//
//        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
//        ov::ParameterVector functionParams {
//            ngraph::builder::makeParams(ngPrc, { {"data", inputDataShape} })[0],
//            ngraph::builder::makeParams(ov::element::i32, { {"indices", indicesShape} })[0]
//        };
//        if (!isAxisConstant) {
//            functionParams.push_back(ngraph::builder::makeParams(ov::element::i32, { {"axis", {1}} })[0]);
//        }
//        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(functionParams));
//        std::shared_ptr<ov::Node> gatherNode;
//        if (isAxisConstant) {
//            gatherNode = std::make_shared<ov::op::v8::Gather>(paramOuts[0], paramOuts[1],
//                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape({}), { axis }), batchDims);
//        } else {
//            gatherNode = std::make_shared<ov::op::v8::Gather>(paramOuts[0], paramOuts[1], paramOuts[2], batchDims);
//        }
//
//        ov::ResultVector results{ std::make_shared<ov::op::v0::Result>(gatherNode) };
//        function = std::make_shared<ov::Model>(results, functionParams, "Gather");
//    }
//
//    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &inputInfo) const override {
//        if (inputInfo.name() == "indices") {
//            const auto& td = inputInfo.getTensorDesc();
//            size_t normAxis = axis < 0 ? axis + dataSrcRank : axis;
//            const auto axDim = targetStaticShapes[index][0][normAxis];
//            if (axDim == 1) {
//                // Random generator cannot generate values in range [0; 0]
//                int values[1] = { 0 };
//                return FuncTestUtils::createAndFillBlobWithFloatArray<int32_t>(td, values, 1);
//            } else {
//                return FuncTestUtils::createAndFillBlob(td, axDim - 1, 0);
//            }
//        } else if (inputInfo.name() == "axis") {
//            int values[1] = { static_cast<int32_t>(axis) };
//            return FuncTestUtils::createAndFillBlobWithFloatArray<int32_t>(inputInfo.getTensorDesc(), values, 1);
//        } else {
//            return LayerTestsCommon::GenerateInput(inputInfo);
//        }
//    }
//
//    int64_t axis = 0;
//    int64_t dataSrcRank = 0;
//};
//
//TEST_P(GatherLayerTestCPU, CompareWithRefs) {
//    SKIP_IF_CURRENT_TEST_IS_DISABLED()
//
//    Run();
//    CheckPluginRelatedResults(executableNetwork, "Gather");
//}
//
//namespace {
//const std::vector<InferenceEngine::Precision> netPrecisions = {
//        InferenceEngine::Precision::FP32,
//        InferenceEngine::Precision::BF16,
//        InferenceEngine::Precision::I8
//};
//
//// 1D
//const std::vector<inputShapesPair> staticInputShapes1D = {
//    {
//        {},
//        { // Static shapes
//            {{4}, {2, 3, 4}}
//        }
//    },
//    {
//        {},
//        { // Static shapes
//            {{4}, {1}}
//        }
//    },
//    {
//        {},
//        { // Static shapes
//            {{4}, {9}}
//        }
//    },
//    {
//        {},
//        { // Static shapes
//            {{5}, {5}}
//        }
//    }
//};
//const std::vector<inputShapesPair> dynamicInputShapes1D = {
//    {
//        { // Origin dynamic shapes
//            {ov::Dimension(4, 6)}, {ov::Dimension(1, 10)},  {ov::Dimension(1, 2)}
//        },
//        { // Dynamic shapes instances
//            {{4}, {1}, {1}},
//            {{4}, {9}, {1}},
//            {{5}, {5}, {1}}
//        }
//    }
//};
//
//INSTANTIATE_TEST_SUITE_P(smoke_StaticShape1D, GatherLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(staticInputShapes1D),
//                    ::testing::Values(0),
//                    ::testing::Values(0),
//                    ::testing::ValuesIn(netPrecisions),
//                    ::testing::Values(true),
//                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                    ::testing::Values(CPUSpecificParams{})),
//                GatherLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape1D, GatherLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(dynamicInputShapes1D),
//                    ::testing::Values(0),
//                    ::testing::Values(0),
//                    ::testing::ValuesIn(netPrecisions),
//                    ::testing::Values(true, false),
//                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                    ::testing::Values(CPUSpecificParams{})),
//                GatherLayerTestCPU::getTestCaseName);
//
//// 2D
//const std::vector<inputShapesPair> staticInputShapes2D = {
//    {
//        {},
//        { // Static shapes
//            {{4, 7}, {4, 55}}
//        }
//    },
//    {
//        {},
//        { // Static shapes
//            {{4, 17}, {4, 17}}
//        }
//    },
//    {
//        {},
//        { // Static shapes
//            {{4, 55}, {4, 7}}
//        }
//    }
//};
//const std::vector<inputShapesPair> dynamicInputShapes2D = {
//    {
//        { // Origin dynamic shapes
//            {4, ov::Dimension(3, 99)},
//            {4, ov::Dimension(3, 99)},
//            {1}
//        },
//        { // Dynamic shapes instances
//            {{4, 7}, {4, 55}, {1}},
//            {{4, 55}, {4, 7}, {1}},
//            {{4, 17}, {4, 17}, {1}}
//        }
//    }
//};
//const std::vector<inputShapesPair> dynamicInputShapes2Dv2 = {
//    {
//        { // Origin dynamic shapes
//            {ov::Dimension(3, 99), ov::Dimension(3, 99)},
//            {-1, ov::Dimension(3, 99)},
//            {1}
//        },
//        { // Dynamic shapes instances
//            {{4, 7}, {4, 55}, {1}},
//            {{8, 55}, {5, 7}, {1}}
//        }
//    }
//};
//
//INSTANTIATE_TEST_SUITE_P(smoke_StaticShape2D, GatherLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(staticInputShapes2D),
//                    ::testing::Values(1),
//                    ::testing::ValuesIn(std::vector<int64_t>{0, 1}),
//                    ::testing::ValuesIn(netPrecisions),
//                    ::testing::Values(true),
//                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                    ::testing::Values(CPUSpecificParams{})),
//                GatherLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape2D, GatherLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(dynamicInputShapes2D),
//                    ::testing::Values(1),
//                    ::testing::ValuesIn(std::vector<int64_t>{0, 1}),
//                    ::testing::ValuesIn(netPrecisions),
//                    ::testing::Values(true, false),
//                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                    ::testing::Values(CPUSpecificParams{})),
//                GatherLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape2Dv2, GatherLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(dynamicInputShapes2Dv2),
//                    ::testing::Values(0),
//                    ::testing::Values(0),
//                    ::testing::ValuesIn(netPrecisions),
//                    ::testing::Values(true, false),
//                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                    ::testing::Values(CPUSpecificParams{})),
//                GatherLayerTestCPU::getTestCaseName);
//
//// 4D
//const std::vector<inputShapesPair> staticInputShapes4D = {
//    {
//        {},
//        { // Static shapes
//            {{4, 5, 6, 7}, {2, 5, 1}}
//        }
//    },
//    {
//        {},
//        { // Static shapes
//            {{10, 5, 6, 7}, {2, 5, 2}}
//        }
//    },
//    {
//        {},
//        { // Static shapes
//            {{16, 5, 6, 7}, {3, 5, 3}}
//        }
//    }
//};
//const std::vector<inputShapesPair> dynamicInputShapes4D = {
//    {
//        { // Origin dynamic shapes
//            {ov::Dimension(4, 20), 5, 6, 7},
//            {ov::Dimension(2, 4), 5, ov::Dimension(1, 4)},
//            {1}
//        },
//        { // Dynamic shapes instances
//            {{4, 5, 6, 7}, {2, 5, 1}, {1}},
//            {{10, 5, 6, 7}, {2, 5, 2}, {1}},
//            {{16, 5, 6, 7}, {3, 5, 3}, {1}}
//        }
//    },
//    {
//        { // Origin dynamic shapes
//            {-1, -1, -1, -1}, {-1, -1, -1}, {1}
//        },
//        { // Dynamic shapes instances
//            {{4, 5, 6, 4}, {2, 5, 16}, {1}},
//            {{10, 5, 6, 8}, {2, 5, 24}, {1}}
//        }
//    }
//};
//
//INSTANTIATE_TEST_SUITE_P(smoke_StaticShape4D, GatherLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(staticInputShapes4D),
//                    ::testing::ValuesIn(std::vector<int64_t>{0, 1, 2, -1}),
//                    ::testing::Values(0),
//                    ::testing::ValuesIn(netPrecisions),
//                    ::testing::Values(true),
//                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                    ::testing::Values(CPUSpecificParams{})),
//                GatherLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape4D, GatherLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(dynamicInputShapes4D),
//                    ::testing::ValuesIn(std::vector<int64_t>{0, 1, 2, -1}),
//                    ::testing::Values(0),
//                    ::testing::ValuesIn(netPrecisions),
//                    ::testing::Values(true, false),
//                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                    ::testing::Values(CPUSpecificParams{})),
//                GatherLayerTestCPU::getTestCaseName);
//} // namespace
//} // namespace CPULayerTestsDefinitions
