// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/fusing_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"

#include <array>
#include <cstdint>

using namespace CPUTestUtils;
namespace ov {
namespace test {

using selectParams = std::tuple<std::vector<InputShape>,    // input shapes
                                ElementType,                // Then/Else precision
                                ov::op::AutoBroadcastSpec,  // broadcast
                                fusingSpecificParams>;

using selectI64ConstParams = std::tuple<InputShape,                // condition input shape
                                        ov::Shape,                 // then constant shape
                                        ov::Shape,                 // else constant shape
                                        ov::op::AutoBroadcastSpec  // broadcast
                                        >;

static std::vector<int64_t> make_i64_values(const ov::Shape& shape, const std::array<int64_t, 4>& values) {
    std::vector<int64_t> data(ov::shape_size(shape));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = values[i % values.size()];
    }
    return data;
}

class SelectLayerCPUTest : public testing::WithParamInterface<selectParams>,
                           virtual public SubgraphBaseTest,
                           public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<selectParams>& obj) {
        const auto& [shapes, precision, broadcast, fusingParams] = obj.param;
        std::ostringstream result;
        result << "Condition_prc_" << ElementType::boolean << "_Then_Else_prc_" << precision << "_";
        result << "IS=(";
        for (const auto& shape : shapes) {
            result << shape.first << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : shapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "Broadcast=" << broadcast.m_type;
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
    void SetUp() override {
        abs_threshold = 0;
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto& [shapes, precision, broadcast, fusingParams] = this->GetParam();
        init_input_shapes(shapes);
        std::tie(inFmts, outFmts, priority, selectedType) = emptyCPUSpec;
        selectedType = precision == ov::element::i64 ? makeSelectedTypeStr("ref", precision)
                                                     : makeSelectedTypeStr(getPrimitiveType(), ov::element::i8);

        ov::element::TypeVector types{ov::element::boolean, precision, precision};
        ov::ParameterVector parameters;
        for (size_t i = 0; i < types.size(); i++) {
            auto param_node = std::make_shared<ov::op::v0::Parameter>(types[i], inputDynamicShapes[i]);
            parameters.push_back(param_node);
        }
        auto select = std::make_shared<ov::op::v1::Select>(parameters[0], parameters[1], parameters[2], broadcast);

        function = create_ov_model(precision, parameters, select, "Eltwise");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();
        if (modelInputs[1].get_element_type() == ov::element::i64) {
            ov::Tensor condTensor{modelInputs[0].get_element_type(), targetInputStaticShapes[0]};
            ov::Tensor thenTensor{modelInputs[1].get_element_type(), targetInputStaticShapes[1]};
            ov::Tensor elseTensor{modelInputs[2].get_element_type(), targetInputStaticShapes[2]};

            auto* condData = condTensor.data<bool>();
            auto* thenData = thenTensor.data<int64_t>();
            auto* elseData = elseTensor.data<int64_t>();
            const std::array<int64_t, 4> thenValues = {(1LL << 45) + 123,
                                                       (1LL << 52) + 7,
                                                       (1LL << 55) + 31,
                                                       (1LL << 60) + 9};
            const std::array<int64_t, 4> elseValues = {(1LL << 44) + 5,
                                                       (1LL << 46) + 11,
                                                       (1LL << 53) + 17,
                                                       (1LL << 54) + 23};
            for (size_t i = 0; i < ov::shape_size(targetInputStaticShapes[0]); ++i) {
                condData[i] = i != 2;
            }
            for (size_t i = 0; i < ov::shape_size(targetInputStaticShapes[1]); ++i) {
                thenData[i] = thenValues[i % thenValues.size()];
            }
            for (size_t i = 0; i < ov::shape_size(targetInputStaticShapes[2]); ++i) {
                elseData[i] = elseValues[i % elseValues.size()];
            }

            inputs.insert({modelInputs[0].get_node_shared_ptr(), condTensor});
            inputs.insert({modelInputs[1].get_node_shared_ptr(), thenTensor});
            inputs.insert({modelInputs[2].get_node_shared_ptr(), elseTensor});
            return;
        }

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -1;
        in_data.range = 3;
        in_data.resolution = 2;
        auto condTensor = ov::test::utils::create_and_fill_tensor(modelInputs[0].get_element_type(), targetInputStaticShapes[0], in_data);

        in_data.start_from = -10;
        in_data.range = 10;
        in_data.resolution = 2;
        auto thenTensor = ov::test::utils::create_and_fill_tensor(modelInputs[1].get_element_type(), targetInputStaticShapes[1], in_data);

        in_data.start_from = 0;
        in_data.range = 10;
        in_data.resolution = 2;
        auto elseTensor = ov::test::utils::create_and_fill_tensor(modelInputs[2].get_element_type(), targetInputStaticShapes[2], in_data);
        inputs.insert({modelInputs[0].get_node_shared_ptr(), condTensor});
        inputs.insert({modelInputs[1].get_node_shared_ptr(), thenTensor});
        inputs.insert({modelInputs[2].get_node_shared_ptr(), elseTensor});
    }
};

TEST_P(SelectLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

class SelectI64ConstLayerCPUTest : public testing::WithParamInterface<selectI64ConstParams>,
                                   virtual public SubgraphBaseTest,
                                   public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<selectI64ConstParams>& obj) {
        const auto& [conditionShape, thenShape, elseShape, broadcast] = obj.param;
        std::ostringstream result;
        result << "Condition_prc_" << ElementType::boolean << "_Then_Else_prc_" << ElementType::i64 << "_";
        result << "Condition_IS=" << conditionShape.first << "_TS=(";
        for (const auto& shape : conditionShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << ")_ThenConst=" << ov::test::utils::vec2str(thenShape) << "_";
        result << "ElseConst=" << ov::test::utils::vec2str(elseShape) << "_";
        result << "Broadcast=" << broadcast.m_type;
        return result.str();
    }

protected:
    void SetUp() override {
        abs_threshold = 0;
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto& [conditionShape, thenShape, elseShape, broadcast] = this->GetParam();
        init_input_shapes({conditionShape});
        selectedType = makeSelectedTypeStr("ref", ov::element::i64);

        auto condition = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, inputDynamicShapes[0]);
        const std::array<int64_t, 4> thenValues = {(1LL << 45) + 123,
                                                   (1LL << 52) + 7,
                                                   (1LL << 55) + 31,
                                                   (1LL << 60) + 9};
        const std::array<int64_t, 4> elseValues = {(1LL << 44) + 5,
                                                   (1LL << 46) + 11,
                                                   (1LL << 53) + 17,
                                                   (1LL << 54) + 23};
        auto thenConstant =
            ov::op::v0::Constant::create(ov::element::i64, thenShape, make_i64_values(thenShape, thenValues));
        auto elseConstant =
            ov::op::v0::Constant::create(ov::element::i64, elseShape, make_i64_values(elseShape, elseValues));
        auto select = std::make_shared<ov::op::v1::Select>(condition, thenConstant, elseConstant, broadcast);

        ov::ParameterVector parameters{condition};
        function = create_ov_model(ov::element::i64, parameters, select, "Eltwise");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();
        ov::Tensor condTensor{modelInputs[0].get_element_type(), targetInputStaticShapes[0]};
        auto* condData = condTensor.data<bool>();
        for (size_t i = 0; i < ov::shape_size(targetInputStaticShapes[0]); ++i) {
            condData[i] = i % 2 == 0;
        }
        inputs.insert({modelInputs[0].get_node_shared_ptr(), condTensor});
    }
};

TEST_P(SelectI64ConstLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

const std::vector<ElementType> precisions = {ElementType::f32, ElementType::i32, ElementType::bf16, ElementType::i8};

const std::vector<fusingSpecificParams> fusingParamsSet{
    emptyFusingSpec,
    fusingSigmoid,
    fusingMultiplyAddPerChannel,
};

const std::vector<std::vector<InputShape>> inShapesDynamicNumpy = {
    {
        // Condition
        {{-1, -1, -1, -1}, {{5, 1, 2, 1}, {1, 1, 1, 1}, {5, 9, 8, 7}}},
        // Then
        {{-1, -1, -1, -1, -1}, {{8, 1, 9, 1, 1}, {1, 1, 1, 1, 1}, {21, 5, 9, 8, 7}}},
        // Else
        {{-1, -1, -1, -1}, {{5, 1, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}},
    },
    {
        // Condition
        {{-1, -1}, {{8, 1}, {10, 5}, {8, 7}}},
        // Then
        {{-1, -1, -1, -1, -1}, {{2, 1, 1, 8, 1}, {7, 8, 3, 10, 5}, {1, 1, 1, 8, 1}}},
        // Else
        {{-1, -1, -1}, {{9, 1, 1}, {3, 10, 5}, {1, 1, 7}}},
    },
    {
        // Condition
        {{{2, 8}, {3, 7}, {1, 10}, {1, 6}, {1, 10}}, {{5, 4, 1, 1, 1}, {8, 5, 5, 5, 1}, {2, 3, 4, 5, 6}}},
        // Then
        {{-1, -1, -1, -1, -1}, {{5, 1, 8, 1, 1}, {8, 1, 1, 1, 8}, {2, 3, 4, 5, 6}}},
        // Else
        {{{1, 5}, {1, 11}, {5, 5}, {1, 8}}, {{1, 1, 5, 1}, {5, 5, 5, 8}, {3, 4, 5, 6}}},
    },
    {
        // Condition
        {{{1, 10}}, {{4}, {10}, {1}}},
        // Then
        {{{1, 15}, {2, 7}, {1, 6}, {5, 12}, {1, 20}}, {{8, 5, 6, 6, 1}, {15, 7, 6, 10, 10}, {2, 5, 4, 5, 3}}},
        // Else
        {{{2, 10}, {1, 16}}, {{6, 4}, {10, 10}, {5, 1}}},
    },
};

const auto numpyCases = ::testing::Combine(::testing::ValuesIn(inShapesDynamicNumpy),
                                           ::testing::ValuesIn(precisions),
                                           ::testing::Values(ov::op::AutoBroadcastType::NUMPY),
                                           ::testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_dynamic,
                         SelectLayerCPUTest,
                         numpyCases,
                         SelectLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inShapesDynamicNone = {
    {
        // Condition
        {{{1, 10}, -1, {10, 20}, {1, 5}}, {{3, 16, 15, 5}, {1, 16, 10, 1}, {10, 16, 20, 5}}},
        // Then
        {{-1, {16, 16}, -1, -1}, {{3, 16, 15, 5}, {1, 16, 10, 1}, {10, 16, 20, 5}}},
        // Else
        {{-1, -1, -1, -1}, {{3, 16, 15, 5}, {1, 16, 10, 1}, {10, 16, 20, 5}}},
    },
};

const auto noneCases = ::testing::Combine(::testing::ValuesIn(inShapesDynamicNone),
                                          ::testing::ValuesIn(precisions),
                                          ::testing::Values(ov::op::AutoBroadcastType::NONE),
                                          ::testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNone_dynamic,
                         SelectLayerCPUTest,
                         noneCases,
                         SelectLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inShapesI64 = {
    {
        {{{4}}, {{4}}},
        {{{4}}, {{4}}},
        {{{4}}, {{4}}},
    },
};

const std::vector<std::vector<InputShape>> inShapesI64Numpy = {
    {
        {{1, 3}, {{1, 3}}},
        {{2, 3}, {{2, 3}}},
        {{1, 1}, {{1, 1}}},
    },
};

const auto i64Cases = ::testing::Combine(::testing::ValuesIn(inShapesI64),
                                         ::testing::Values(ElementType::i64),
                                         ::testing::Values(ov::op::AutoBroadcastType::NONE),
                                         ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_i64,
                         SelectLayerCPUTest,
                         i64Cases,
                         SelectLayerCPUTest::getTestCaseName);

const auto i64NumpyCases = ::testing::Combine(::testing::ValuesIn(inShapesI64Numpy),
                                              ::testing::Values(ElementType::i64),
                                              ::testing::Values(ov::op::AutoBroadcastType::NUMPY),
                                              ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_i64_Broadcast,
                         SelectLayerCPUTest,
                         i64NumpyCases,
                         SelectLayerCPUTest::getTestCaseName);

const auto i64ConstCases = ::testing::Combine(::testing::Values(InputShape{{1, 3}, {{1, 3}}}),
                                             ::testing::Values(ov::Shape{2, 3}),
                                             ::testing::Values(ov::Shape{1, 1}),
                                             ::testing::Values(ov::op::AutoBroadcastType::NUMPY));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_i64_ConstBroadcast,
                         SelectI64ConstLayerCPUTest,
                         i64ConstCases,
                         SelectI64ConstLayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
