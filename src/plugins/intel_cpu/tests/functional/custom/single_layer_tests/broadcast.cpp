// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using BroadcastLayerTestParamsSet = typename std::tuple<std::vector<ov::test::InputShape>,  // Shapes
                                                        std::vector<int64_t>,               // Target shapes
                                                        std::vector<int64_t>,               // Axes mapping
                                                        ov::op::BroadcastType,              // Broadcast mode
                                                        ov::element::Type_t,                // Network precision
                                                        std::vector<bool>,                  // Const inputs
                                                        std::string>;                       // Device name

using BroadcastLayerCPUTestParamsSet = typename std::tuple<BroadcastLayerTestParamsSet, CPUSpecificParams>;

class BroadcastLayerCPUTest : public testing::WithParamInterface<BroadcastLayerCPUTestParamsSet>,
                              virtual public ov::test::SubgraphBaseTest,
                              public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BroadcastLayerCPUTestParamsSet> obj) {
        BroadcastLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::vector<ov::test::InputShape> inputShapes;
        std::vector<int64_t> targetShapes, axesMapping;
        ov::op::BroadcastType mode;
        ov::element::Type_t netPrecision;
        std::vector<bool> isConstInputs;
        std::string deviceName;
        std::tie(inputShapes, targetShapes, axesMapping, mode, netPrecision, isConstInputs, deviceName) =
            basicParamsSet;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "targetShape=" << ov::test::utils::vec2str(targetShapes) << "_";
        result << "axesMapping=" << ov::test::utils::vec2str(axesMapping) << "_";
        result << "mode=" << mode << "_";
        result << "netPrec=" << netPrecision << "_";
        result << "constIn=(" << (isConstInputs[0] ? "True" : "False") << "." << (isConstInputs[1] ? "True" : "False")
               << ")_";
        result << "trgDev=" << deviceName;

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        BroadcastLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::vector<ov::test::InputShape> inputShapes;
        ov::op::BroadcastType mode;
        ov::element::Type_t netPrecision;
        std::vector<bool> isConstInput;
        std::tie(inputShapes, targetShape, axesMapping, mode, netPrecision, isConstInput, targetDevice) =
            basicParamsSet;
        bool isTargetShapeConst = isConstInput[0], isAxesMapConst = isConstInput[1];
        const auto targetShapeRank = targetShape.size();
        const auto axesMappingRank = axesMapping.size();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        selectedType += std::string("_") + ov::element::Type(netPrecision).get_type_name();

        if (inputShapes.front().first.rank() != 0) {
            inputDynamicShapes.push_back(inputShapes.front().first);
            if (!isTargetShapeConst) {
                inputDynamicShapes.push_back({static_cast<int64_t>(targetShape.size())});
            }
            if (!isAxesMapConst) {
                inputDynamicShapes.push_back({static_cast<int64_t>(axesMapping.size())});
            }
        }
        const size_t targetStaticShapeSize = inputShapes.front().second.size();
        targetStaticShapes.resize(targetStaticShapeSize);
        for (size_t i = 0lu; i < targetStaticShapeSize; ++i) {
            targetStaticShapes[i].push_back(inputShapes.front().second[i]);
            if (!isTargetShapeConst)
                targetStaticShapes[i].push_back({targetShape.size()});
            if (!isAxesMapConst)
                targetStaticShapes[i].push_back({axesMapping.size()});
        }

        ov::ParameterVector functionParams;
        if (inputDynamicShapes.empty()) {
            functionParams.push_back(
                std::make_shared<ov::op::v0::Parameter>(netPrecision, targetStaticShapes.front().front()));
        } else {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes.front()));
            if (!isTargetShapeConst) {
                functionParams.push_back(
                    std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[1]));
                functionParams.back()->set_friendly_name("targetShape");
            }
            if (!isAxesMapConst) {
                functionParams.push_back(
                    std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes.back()));
                functionParams.back()->set_friendly_name("axesMapping");
            }
        }
        functionParams.front()->set_friendly_name("data");

        std::shared_ptr<ov::op::v3::Broadcast> broadcastOp;
        if (mode == ov::op::BroadcastType::EXPLICIT) {
            std::shared_ptr<ov::Node> targetShapeOp;
            std::shared_ptr<ov::Node> axesMappingOp;
            if (isTargetShapeConst) {
                targetShapeOp = ov::op::v0::Constant::create(ov::element::i64, {targetShapeRank}, targetShape);
            } else {
                targetShapeOp = functionParams[0];
            }
            if (isAxesMapConst) {
                axesMappingOp = ov::op::v0::Constant::create(ov::element::i64, {axesMappingRank}, axesMapping);
            } else {
                axesMappingOp = functionParams.size() > 2 ? functionParams[2] : functionParams[1];
            }
            broadcastOp =
                std::make_shared<ov::op::v3::Broadcast>(functionParams[0], targetShapeOp, axesMappingOp, mode);
        } else if (mode == ov::op::BroadcastType::NUMPY) {
            if (isTargetShapeConst) {
                auto targetShapeConst = ov::op::v0::Constant::create(ov::element::i64, {targetShapeRank}, targetShape);
                broadcastOp = std::make_shared<ov::op::v3::Broadcast>(functionParams[0], targetShapeConst, mode);
            } else {
                broadcastOp = std::make_shared<ov::op::v3::Broadcast>(functionParams[0], functionParams[1], mode);
            }
        }

        function = makeNgraphFunction(netPrecision, functionParams, broadcastOp, "Broadcast");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0lu; i < funcInputs.size(); i++) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (funcInput.get_node()->get_friendly_name() == "targetShape") {
                tensor = ov::Tensor{ov::element::i64, targetInputStaticShapes[i]};
                auto data = tensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
                for (size_t i = 0lu; i < targetShape.size(); i++) {
                    data[i] = targetShape[i];
                }
            } else if (funcInput.get_node()->get_friendly_name() == "axesMapping") {
                tensor = ov::Tensor{ov::element::i64, targetInputStaticShapes[i]};
                auto data = tensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
                for (size_t i = 0lu; i < axesMapping.size(); i++) {
                    data[i] = axesMapping[i];
                }
            } else {
                if (funcInput.get_element_type().is_real()) {
                    ov::test::utils::InputGenerateData in_data;
                    in_data.start_from = 0;
                    in_data.range = 10;
                    in_data.resolution = 1000;
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
                } else {
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                     targetInputStaticShapes[i]);
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    std::vector<int64_t> targetShape, axesMapping;
};

TEST_P(BroadcastLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Broadcast");
}

namespace {

/* CPU PARAMS */
const auto cpuParams_nChw16c = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, "ref"};
const auto cpuParams_nCdhw16c = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, "ref"};

const auto cpuParams_nChw8c = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, "ref"};
const auto cpuParams_nCdhw8c = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, "ref"};

const auto cpuParams_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {}, "ref"};
const auto cpuParams_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, "ref"};
/* ========== */

/* COMMON PARAMS */
const std::vector<ov::element::Type_t> inputPrecisions = {ov::element::f32,
                                                          ov::element::bf16,
                                                          ov::element::i32,
                                                          ov::element::i8};
/* ============= */

/* INSTANCES */
// 4D
const std::vector<CPUSpecificParams> CPUParams4D = {cpuParams_nChw16c, cpuParams_nChw8c, cpuParams_nhwc};

const std::vector<std::vector<ov::test::InputShape>> staticInputShapes4D = {{{{},
                                                                              {// Static shapes
                                                                               {1, 16, 1, 1}}}},
                                                                            {{{},
                                                                              {// Static shapes
                                                                               {50, 50}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_StaticShape4D,
                        BroadcastLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::Values(staticInputShapes4D[0]),
                                                              ::testing::ValuesIn(std::vector<std::vector<int64_t>>{
                                                                  {1, 16, 3, 3},
                                                                  {1, 16, 1, 3}}),
                                                              ::testing::Values(std::vector<int64_t>{}),
                                                              ::testing::Values(ov::op::BroadcastType::NUMPY),
                                                              ::testing::ValuesIn(inputPrecisions),
                                                              ::testing::Values(std::vector<bool>{true, true}),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::ValuesIn(CPUParams4D)),
                        BroadcastLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticShape4DE,
                        BroadcastLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::Values(staticInputShapes4D[1]),
                                                              ::testing::Values(std::vector<int64_t>{1, 50, 50, 16}),
                                                              ::testing::Values(std::vector<int64_t>{1, 2}),
                                                              ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                                                              ::testing::ValuesIn(inputPrecisions),
                                                              ::testing::Values(std::vector<bool>{true, true}),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        BroadcastLayerCPUTest::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> staticInputShapesScalar = {{{{},
                                                                                  {// Static shapes
                                                                                   {1}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_StaticShape4DScalar,
                        BroadcastLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(staticInputShapesScalar),
                                                              ::testing::Values(std::vector<int64_t>{1, 16, 3, 3}),
                                                              ::testing::Values(std::vector<int64_t>{}),
                                                              ::testing::Values(ov::op::BroadcastType::NUMPY),
                                                              ::testing::ValuesIn(inputPrecisions),
                                                              ::testing::Values(std::vector<bool>{true, true}),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        BroadcastLayerCPUTest::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes4D = {
    {
        {// Origin dynamic shapes
         {ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20)},
         {// Dynamic shapes instances
          {1, 16, 1, 1},
          {8, 1, 1, 7},
          {1, 1, 1, 7}}},
    },
    {{// Origin dynamic shapes
      {-1, -1, -1, -1},
      {// Dynamic shapes instances
       {{1, 16, 1, 1}},
       {{8, 1, 1, 1}}}}}};

INSTANTIATE_TEST_SUITE_P(
    smoke_DynamicShape4D,
    BroadcastLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(dynamicInputShapes4D),
                           ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{8, 16, 1, 7}, {8, 16, 10, 7}}),
                           ::testing::Values(std::vector<int64_t>{}),
                           ::testing::Values(ov::op::BroadcastType::NUMPY),
                           ::testing::ValuesIn(inputPrecisions),
                           ::testing::ValuesIn(std::vector<std::vector<bool>>{{true, true}, {false, true}}),
                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
    BroadcastLayerCPUTest::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapesScalar = {{{// Origin dynamic shapes
                                                                                   {-1},
                                                                                   {// Dynamic shapes instances
                                                                                    {1},
                                                                                    {7}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape4DScalar,
                        BroadcastLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(dynamicInputShapesScalar),
                                                              ::testing::Values(std::vector<int64_t>{8, 16, 1, 7}),
                                                              ::testing::Values(std::vector<int64_t>{}),
                                                              ::testing::Values(ov::op::BroadcastType::NUMPY),
                                                              ::testing::ValuesIn(inputPrecisions),
                                                              ::testing::ValuesIn(std::vector<std::vector<bool>>{
                                                                  {true, true},
                                                                  {false, true}}),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        BroadcastLayerCPUTest::getTestCaseName);

// 5D
const std::vector<std::vector<ov::test::InputShape>> staticInputShapes5D = {{{{},
                                                                              {// Static shapes
                                                                               {1, 16, 1, 1, 1}}}}};
const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes5D = {
    {{// Origin dynamic shapes
      {ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20)},
      {// Dynamic shapes instances
       {1, 16, 1, 1, 1},
       {8, 1, 1, 7, 1},
       {8, 1, 1, 1, 1}}}},
    {{// Origin dynamic shapes
      {-1, -1, -1, -1, -1},
      {// Dynamic shapes instances
       {1, 16, 1, 1, 1},
       {8, 16, 1, 7, 1}}}}};
std::vector<std::vector<int64_t>> targetShapes5D{{8, 16, 1, 7, 1}, {8, 16, 10, 7, 4}};

const std::vector<CPUSpecificParams> CPUParams5D = {
    cpuParams_nCdhw16c,
    cpuParams_nCdhw8c,
    cpuParams_ndhwc,
};

INSTANTIATE_TEST_SUITE_P(smoke_StaticShape5D,
                        BroadcastLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(staticInputShapes5D),
                                                              ::testing::ValuesIn(std::vector<std::vector<int64_t>>{
                                                                  {1, 16, 1, 1, 3},
                                                                  {1, 16, 3, 1, 3}}),
                                                              ::testing::Values(std::vector<int64_t>{}),
                                                              ::testing::Values(ov::op::BroadcastType::NUMPY),
                                                              ::testing::ValuesIn(inputPrecisions),
                                                              ::testing::Values(std::vector<bool>{true, true}),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::ValuesIn(CPUParams5D)),
                        BroadcastLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticShape5DScalar,
                        BroadcastLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(staticInputShapesScalar),
                                                              ::testing::Values(std::vector<int64_t>{1, 16, 3, 1, 3}),
                                                              ::testing::Values(std::vector<int64_t>{}),
                                                              ::testing::Values(ov::op::BroadcastType::NUMPY),
                                                              ::testing::ValuesIn(inputPrecisions),
                                                              ::testing::Values(std::vector<bool>{true, true}),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        BroadcastLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape5D,
                        BroadcastLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(dynamicInputShapes5D),
                                                              ::testing::ValuesIn(targetShapes5D),
                                                              ::testing::Values(std::vector<int64_t>{}),
                                                              ::testing::Values(ov::op::BroadcastType::NUMPY),
                                                              ::testing::ValuesIn(inputPrecisions),
                                                              ::testing::ValuesIn(std::vector<std::vector<bool>>{
                                                                  {true, true},
                                                                  {false, true}}),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        BroadcastLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape5DScalar,
                        BroadcastLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(dynamicInputShapesScalar),
                                                              ::testing::Values(std::vector<int64_t>{8, 16, 1, 1, 7}),
                                                              ::testing::Values(std::vector<int64_t>{}),
                                                              ::testing::Values(ov::op::BroadcastType::NUMPY),
                                                              ::testing::ValuesIn(inputPrecisions),
                                                              ::testing::ValuesIn(std::vector<std::vector<bool>>{
                                                                  {true, true},
                                                                  {false, true}}),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        BroadcastLayerCPUTest::getTestCaseName);

// 1D
const std::vector<std::vector<ov::test::InputShape>> dynamicShapes1D = {{{// Origin dynamic shapes
                                                                          {-1},
                                                                          {// Dynamic shapes instances
                                                                           {1},
                                                                           {1}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapes1D,
                        BroadcastLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(dynamicShapes1D),
                                                              ::testing::Values(std::vector<int64_t>{0}),
                                                              ::testing::Values(std::vector<int64_t>{}),
                                                              ::testing::Values(ov::op::BroadcastType::NUMPY),
                                                              ::testing::ValuesIn(inputPrecisions),
                                                              ::testing::ValuesIn(std::vector<std::vector<bool>>{
                                                                  {false, true}}),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        BroadcastLayerCPUTest::getTestCaseName);
/* ========= */

}  // namespace

}  // namespace test
}  // namespace ov
