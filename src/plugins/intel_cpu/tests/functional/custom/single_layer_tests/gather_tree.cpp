// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/test_enums.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using GatherTreeCPUTestParams = typename std::tuple<InputShape,                       // Input tensors shape
                                                    ov::test::utils::InputLayerType,  // Secondary input type
                                                    ov::element::Type,                // Network precision
                                                    ov::element::Type,                // Input precision
                                                    ov::element::Type,                // Output precision
                                                    std::string>;                     // Device name

class GatherTreeLayerCPUTest : public testing::WithParamInterface<GatherTreeCPUTestParams>,
                               virtual public ov::test::SubgraphBaseTest,
                               public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherTreeCPUTestParams>& obj) {
        InputShape inputShape;
        ov::element::Type netPrecision;
        ov::element::Type inPrc, outPrc;
        ov::test::utils::InputLayerType secondaryInputType;
        std::string targetName;

        std::tie(inputShape, secondaryInputType, netPrecision, inPrc, outPrc, targetName) = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& item : inputShape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "secondaryInputType=" << secondaryInputType << "_";
        result << "netPRC=" << netPrecision.get_type_name() << "_";
        result << "inPRC=" << inPrc.get_type_name() << "_";
        result << "outPRC=" << outPrc.get_type_name() << "_";
        result << "trgDev=" << targetName;

        return result.str();
    }

protected:
    void SetUp() override {
        InputShape inputShape;
        ov::element::Type netPrecision;
        ov::test::utils::InputLayerType secondaryInputType;
        ov::element::Type inPrc, outPrc;

        std::tie(inputShape, secondaryInputType, netPrecision, inPrc, outPrc, targetDevice) = GetParam();
        InputShape parentShape{inputShape};
        InputShape::first_type maxSeqLenFirst;
        if (inputShape.first.is_dynamic()) {
            maxSeqLenFirst = {inputShape.first[1]};
        }
        InputShape::second_type maxSeqLenSecond;
        maxSeqLenSecond.reserve(inputShape.second.size());
        for (const auto& item : inputShape.second) {
            maxSeqLenSecond.emplace_back(std::initializer_list<size_t>{item[1]});
        }
        InputShape maxSeqLenShape{std::move(maxSeqLenFirst), std::move(maxSeqLenSecond)};

        init_input_shapes({inputShape, parentShape, maxSeqLenShape});

        // initialization of scalar input as it cannot be done properly in init_input_shapes
        inputDynamicShapes.push_back({});
        for (auto& shape : targetStaticShapes) {
            shape.push_back({});
        }

        std::shared_ptr<ov::Node> inp2;
        std::shared_ptr<ov::Node> inp3;
        std::shared_ptr<ov::Node> inp4;

        ov::ParameterVector paramsIn{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0])};
        if (ov::test::utils::InputLayerType::PARAMETER == secondaryInputType) {
            auto param2 = std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[1]);
            auto param3 = std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[2]);
            auto param4 = std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[3]);
            inp2 = param2;
            inp3 = param3;
            inp4 = param4;

            paramsIn.push_back(param2);
            paramsIn.push_back(param3);
            paramsIn.push_back(param4);
        } else if (ov::test::utils::InputLayerType::CONSTANT == secondaryInputType) {
            auto maxBeamIndex = inputShape.second.front().at(2) - 1;
            ov::test::utils::InputGenerateData in_gen_data(1, maxBeamIndex - 1);
            inp2 = ov::test::utils::make_constant(netPrecision, inputShape.second.front(), in_gen_data);
            inp3 = ov::test::utils::make_constant(netPrecision, {inputShape.second.front().at(1)}, in_gen_data);
            inp4 = ov::test::utils::make_constant(netPrecision, {}, in_gen_data);
        } else {
            throw std::runtime_error("Unsupported inputType");
        }

        auto operationResult = std::make_shared<ov::op::v1::GatherTree>(paramsIn.front(), inp2, inp3, inp4);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(operationResult)};
        function = std::make_shared<ov::Model>(results, paramsIn, "GatherTree");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto maxBeamIndex = targetInputStaticShapes.front().at(2) - 1;
        const auto& funcInputs = function->inputs();

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = (i == 2 || i == 3) ? maxBeamIndex / 2 : 0;
            in_data.range = maxBeamIndex;
            auto tensor = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(GatherTreeLayerCPUTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ov::element::Type> netPrecisions = {ov::element::f32, ov::element::i32};

const std::vector<InputShape> inputStaticShapes = {{{}, {{5, 1, 10}}},
                                                   {{}, {{1, 1, 10}}},
                                                   {{}, {{20, 1, 10}}},
                                                   {{}, {{20, 20, 10}}}};

const std::vector<InputShape> inputDynamicShapesParameter = {{{-1, 1, -1}, {{7, 1, 10}, {8, 1, 20}}},
                                                             {{-1, 1, {5, 10}}, {{2, 1, 7}, {5, 1, 8}}},
                                                             {{-1, {1, 5}, 10}, {{20, 1, 10}, {17, 2, 10}}},
                                                             {{-1, -1, -1}, {{20, 20, 15}, {30, 30, 10}}}};

const std::vector<InputShape> inputDynamicShapesConstant = {{{-1, 1, -1}, {{7, 1, 10}}},
                                                            {{-1, 1, {5, 10}}, {{2, 1, 7}}},
                                                            {{-1, {1, 5}, 10}, {{20, 1, 10}}},
                                                            {{-1, -1, -1}, {{20, 20, 15}}}};

const std::vector<ov::test::utils::InputLayerType> secondaryInputTypes = {ov::test::utils::InputLayerType::CONSTANT,
                                                                          ov::test::utils::InputLayerType::PARAMETER};

INSTANTIATE_TEST_SUITE_P(smoke_GatherTreeCPUStatic,
                         GatherTreeLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputStaticShapes),
                                            ::testing::ValuesIn(secondaryInputTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatherTreeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherTreeCPUDynamicParameter,
                         GatherTreeLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputDynamicShapesParameter),
                                            ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatherTreeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherTreeCPUDynamicConstant,
                         GatherTreeLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputDynamicShapesConstant),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatherTreeLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
