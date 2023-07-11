// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using selectParams = std::tuple<std::vector<InputShape>,         // input shapes
                                ElementType,                     // Then/Else precision
                                ngraph::op::AutoBroadcastSpec,   // broadcast
                                fusingSpecificParams>;

class SelectLayerCPUTest : public testing::WithParamInterface<selectParams>,
                           virtual public SubgraphBaseTest,
                           public CpuTestWithFusing {
public:
   static std::string getTestCaseName(testing::TestParamInfo<selectParams> obj) {
       std::vector<InputShape> shapes;
       ElementType precision;
       ngraph::op::AutoBroadcastSpec broadcast;
       fusingSpecificParams fusingParams;
       std::tie(shapes, precision, broadcast, fusingParams) = obj.param;

       std::ostringstream result;
       result << "Condition_prc_" << ElementType::boolean << "_Then_Else_prc_" << precision << "_";
       result << "IS=(";
       for (const auto& shape : shapes) {
           result << shape.first << "_";
       }
       result << ")_TS=(";
       for (const auto& shape : shapes) {
           for (const auto& item : shape.second) {
               result << CommonTestUtils::vec2str(item) << "_";
           }
       }
       result << "Broadcast=" << broadcast.m_type;
       result << CpuTestWithFusing::getTestCaseName(fusingParams);

       return result.str();
   }

protected:
   void SetUp() override {
        abs_threshold = 0;
        targetDevice = CommonTestUtils::DEVICE_CPU;
        std::vector<InputShape> shapes;
        ElementType precision;
        ngraph::op::AutoBroadcastSpec broadcast;
        fusingSpecificParams fusingParams;
        std::tie(shapes, precision, broadcast, fusingParams) = this->GetParam();
        init_input_shapes(shapes);
        std::tie(inFmts, outFmts, priority, selectedType) = emptyCPUSpec;
        selectedType = makeSelectedTypeStr(getPrimitiveType(), ov::element::i8);

        auto parameters = ngraph::builder::makeDynamicParams(ov::element::TypeVector{ov::element::boolean, precision, precision}, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(parameters));
        auto select = ngraph::builder::makeSelect(paramOuts, broadcast);

        function = makeNgraphFunction(precision, parameters, select, "Eltwise");
   }

   void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();
        auto condTensor = ov::test::utils::create_and_fill_tensor(modelInputs[0].get_element_type(), targetInputStaticShapes[0], 3, -1, 2);
        auto thenTensor = ov::test::utils::create_and_fill_tensor(modelInputs[1].get_element_type(), targetInputStaticShapes[1], 10, -10, 2);
        auto elseTensor = ov::test::utils::create_and_fill_tensor(modelInputs[2].get_element_type(), targetInputStaticShapes[2], 10, 0, 2);
        inputs.insert({modelInputs[0].get_node_shared_ptr(), condTensor});
        inputs.insert({modelInputs[1].get_node_shared_ptr(), thenTensor});
        inputs.insert({modelInputs[2].get_node_shared_ptr(), elseTensor});
    }
};

TEST_P(SelectLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Eltwise");
}

const std::vector<ElementType> precisions = {
    ElementType::f32,
    ElementType::i32,
    ElementType::bf16,
    ElementType::i8
};

const std::vector<fusingSpecificParams> fusingParamsSet{
    emptyFusingSpec,
    fusingSigmoid,
    fusingMultiplyAddPerChannel,
};

const std::vector<std::vector<InputShape>> inShapesDynamicNumpy = {
    {
        // Condition
        {
            {-1, -1, -1, -1},
            {{5, 1, 2, 1}, {1, 1, 1, 1}, {5, 9, 8, 7}}
        },
        // Then
        {
            {-1, -1, -1, -1, -1},
            {{8, 1, 9, 1, 1}, {1, 1, 1, 1, 1}, {21, 5, 9, 8, 7}}
        },
        // Else
        {
            {-1, -1, -1, -1},
            {{5, 1, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}
        },
    },
    {
        // Condition
        {
            {-1, -1},
            {{8, 1}, {10, 5}, {8, 7}}
        },
        // Then
        {
            {-1, -1, -1, -1, -1},
            {{2, 1, 1, 8, 1}, {7, 8, 3, 10, 5}, {1, 1, 1, 8, 1}}
        },
        // Else
        {
            {-1, -1, -1},
            {{9, 1, 1}, {3, 10, 5}, {1, 1, 7}}
        },
    },
    {
        // Condition
        {
            {{2, 8}, {3, 7}, {1, 10}, {1, 6}, {1, 10}},
            {{5, 4, 1, 1, 1}, {8, 5, 5, 5, 1}, {2, 3, 4, 5, 6}}
        },
        // Then
        {
            {-1, -1, -1, -1, -1},
            {{5, 1, 8, 1, 1}, {8, 1, 1, 1, 8}, {2, 3, 4, 5, 6}}
        },
        // Else
        {
            {{1, 5}, {1, 11}, {5, 5}, {1, 8}},
            {{1, 1, 5, 1}, {5, 5, 5, 8}, {3, 4, 5, 6}}
        },
    },
    {
        // Condition
        {
            {{1, 10}},
            {{4}, {10}, {1}}
        },
        // Then
        {
            {{1, 15}, {2, 7}, {1, 6}, {5, 12}, {1, 20}},
            {{8, 5, 6, 6, 1}, {15, 7, 6, 10, 10}, {2, 5, 4, 5, 3}}
        },
        // Else
        {
            {{2, 10}, {1, 16}},
            {{6, 4}, {10, 10}, {5, 1}}
        },
    },
};

const auto numpyCases = ::testing::Combine(::testing::ValuesIn(inShapesDynamicNumpy),
                                           ::testing::ValuesIn(precisions),
                                           ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY),
                                           ::testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_dynamic, SelectLayerCPUTest, numpyCases, SelectLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inShapesDynamicNone = {
    {
        // Condition
        {
            {{1, 10}, -1, {10, 20}, {1, 5}},
            {{3, 16, 15, 5}, {1, 16, 10, 1}, {10, 16, 20, 5}}
        },
        // Then
        {
            {-1, {16, 16}, -1, -1},
            {{3, 16, 15, 5}, {1, 16, 10, 1}, {10, 16, 20, 5}}
        },
        // Else
        {
            {-1, -1, -1, -1},
            {{3, 16, 15, 5}, {1, 16, 10, 1}, {10, 16, 20, 5}}
        },
    },
};

const auto noneCases = ::testing::Combine(::testing::ValuesIn(inShapesDynamicNone),
                                          ::testing::ValuesIn(precisions),
                                          ::testing::Values(ngraph::op::AutoBroadcastType::NONE),
                                          ::testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNone_dynamic, SelectLayerCPUTest, noneCases, SelectLayerCPUTest::getTestCaseName);


class SelectLayerCPUStaticTest : public testing::WithParamInterface<selectParams>,
                           virtual public SubgraphBaseTest,
                           public CpuTestWithFusing {
public:
   static std::string getTestCaseName(testing::TestParamInfo<selectParams> obj) {
        return "EltwiseSelectStaticTest";
   }

protected:
   void SetUp() override {
        abs_threshold = 0;
        targetDevice = CommonTestUtils::DEVICE_CPU;
        std::vector<InputShape> shapes;
        ElementType precision;
        ngraph::op::AutoBroadcastSpec broadcast;
        fusingSpecificParams fusingParams;
        std::tie(shapes, precision, broadcast, fusingParams) = this->GetParam();
        init_input_shapes(shapes);
        std::tie(inFmts, outFmts, priority, selectedType) = emptyCPUSpec;
        selectedType = makeSelectedTypeStr(getPrimitiveType(), ov::element::i8);

        auto parameters = ngraph::builder::makeDynamicParams(ov::element::TypeVector{ov::element::boolean, precision, precision}, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(parameters));
        auto select = ngraph::builder::makeSelect(paramOuts, broadcast);

        function = makeNgraphFunction(precision, parameters, select, "Eltwise");
   }

    ov::Tensor create_and_fill_tensor_static(
            const ov::element::Type element_type,
            const ov::Shape& shape,
            const size_t port) {
        auto tensor = ov::Tensor{element_type, shape};
        if (port == 0) {
            static_cast<bool*>(tensor.data())[0] = false;
            static_cast<bool*>(tensor.data())[1] = true;
        } else if (port == 1) {
            //static_cast<int64_t*>(tensor.data())[0] = -2147483648;
            static_cast<int64_t*>(tensor.data())[0] = -1073741824;
            static_cast<int64_t*>(tensor.data())[1] = 18;
        } else if (port == 2) {
            static_cast<int64_t*>(tensor.data())[0] = 2147483648;
            //static_cast<int64_t*>(tensor.data())[0] = 1073741824;
            static_cast<int64_t*>(tensor.data())[1] = -1;
        }
        return tensor;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();
        auto condTensor = create_and_fill_tensor_static(modelInputs[0].get_element_type(), targetInputStaticShapes[0], 0);
        auto thenTensor = create_and_fill_tensor_static(modelInputs[1].get_element_type(), targetInputStaticShapes[1], 1);
        auto elseTensor = create_and_fill_tensor_static(modelInputs[2].get_element_type(), targetInputStaticShapes[2], 2);
        inputs.insert({modelInputs[0].get_node_shared_ptr(), condTensor});
        inputs.insert({modelInputs[1].get_node_shared_ptr(), thenTensor});
        inputs.insert({modelInputs[2].get_node_shared_ptr(), elseTensor});
    }
};

TEST_P(SelectLayerCPUStaticTest, CompareWithRefsStatic) {
    run();
    CheckPluginRelatedResults(compiledModel, "Eltwise");
}

const std::vector<std::vector<InputShape>> inShapesStatic = {
    {
        // Condition
        {
            {-1},
            {{2}}
        },
        // Then
        {
            {-1},
            {{2}}
        },
        // Else
        {
            {-1},
            {{2}}
        },
    },
};

const std::vector<fusingSpecificParams> fusingParamsSetStatic{
    emptyFusingSpec
};

const std::vector<ElementType> precisionsStatic = {
    ElementType::i64
};

const auto staticCases = ::testing::Combine(::testing::ValuesIn(inShapesStatic),
                                          ::testing::ValuesIn(precisionsStatic),
                                          ::testing::Values(ngraph::op::AutoBroadcastType::NONE),
                                          ::testing::ValuesIn(fusingParamsSetStatic));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNone_static, SelectLayerCPUStaticTest, staticCases, SelectLayerCPUStaticTest::getTestCaseName);
} // namespace CPULayerTestsDefinitions
