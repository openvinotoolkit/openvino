// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

/*
The main purpose of the tests is to test cyclic inplace resolution in order to make sure that output edges are referenced whenever possible.
*/

using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {

using VectorShapes = std::vector<InputShape>;

class SoftmaxAddReshapeOutputSubgraphTest : virtual public ov::test::SubgraphBaseTest {
/*This test runs the following subgraph:

                      param
                        |
                        |
                      Softmax
                     /     \
                    /       \
                   Add     Reshape0
                    |         |
                    |         |
                 Result0   Result1

expect edge Reshape1->Result1 to be referenced by its upstreams, instead of referencing to its upstreams.
*/
protected:
    void SetUp() override {
        constexpr size_t softmax_axis = 1ul;
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto precision = ov::element::f32;
        ov::test::InputShape input_shape{{-1, 16}, {{3, 16}}};
        init_input_shapes({input_shape});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, shape));
        }
        auto soft_max = std::make_shared<ngraph::opset1::Softmax>(params.front(), softmax_axis);

        auto add_const = ngraph::builder::makeConstant(precision, {1}, std::vector<float>({1.0f}));
        auto add_0 = ngraph::builder::makeEltwise(soft_max, add_const, ngraph::helpers::EltwiseTypes::ADD);
        auto result_0 = std::make_shared<ngraph::opset3::Result>(add_0);    // dummy output

        auto reshape_param_0 = ngraph::builder::makeConstant<int>(ov::element::i32, {1}, {0});
        auto reshape_0 = std::make_shared<ngraph::opset1::Unsqueeze>(soft_max, reshape_param_0);
        auto result_1 = std::make_shared<ngraph::opset3::Result>(reshape_0);    // target output

        ngraph::ResultVector results = {result_0, result_1};
        function = std::make_shared<ov::Model>(results, params, "Subgraph1");

        ov::pass::Serialize serializer("Subgraph1.xml", "Subgraph1.bin");
        serializer.run_on_model(function);
    }
};

TEST_F(SoftmaxAddReshapeOutputSubgraphTest, smoke_CompareWithRefs) {
    run();
}


class SoftmaxAddReshapeTwoOutputsSubgraphTest : public testing::WithParamInterface<VectorShapes>,
                                                virtual public ov::test::SubgraphBaseTest {
/*This test runs the following subgraph:

                      param
                        |
                        |
                      Softmax
                     /       \
                    /         \
                   Add       Reshape0
                    |         |      \
                    |         |       \
                 Result0   Reshape1  Result2
                              |         
                              |
                            Result1

Edge Reshape1 -> Result1 cannot be referenced by its upstreams as there are more than one outputs referencing it.
*/
public:
    static std::string getTestCaseName(testing::TestParamInfo<VectorShapes> obj) {
        VectorShapes& inputShapes = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")";
        }
        return result.str();
    }

    void SetUp() override {
        constexpr size_t softmax_axis = 1ul;
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto precision = ov::element::f32;
        auto& input_shape = this->GetParam();
        init_input_shapes({input_shape});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, shape));
        }
        auto soft_max = std::make_shared<ngraph::opset1::Softmax>(params.front(), softmax_axis);

        auto add_const = ngraph::builder::makeConstant(precision, {1}, std::vector<float>({1.0f}));
        auto add_0 = ngraph::builder::makeEltwise(soft_max, add_const, ngraph::helpers::EltwiseTypes::ADD);
        auto result_0 = std::make_shared<ngraph::opset3::Result>(add_0);    // dummy output

        auto reshape_param_0 = ngraph::builder::makeConstant<int>(ov::element::i32, {1}, {0});
        auto reshape_0 = std::make_shared<ngraph::opset1::Unsqueeze>(soft_max, reshape_param_0);
        auto result_2 = std::make_shared<ngraph::opset3::Result>(reshape_0);    // dummy output

        auto reshape_param_1 = ngraph::builder::makeConstant<int>(ov::element::i32, {1}, {0});
        auto reshape_1 = std::make_shared<ngraph::opset1::Unsqueeze>(reshape_0, reshape_param_1);
        auto result_1 = std::make_shared<ngraph::opset3::Result>(reshape_1);    // target output

        ngraph::ResultVector results = {result_0, result_1, result_2};
        function = std::make_shared<ov::Model>(results, params, "Subgraph2");

        ov::pass::Serialize serializer("Subgraph2.xml", "Subgraph2.bin");
        serializer.run_on_model(function);
    }
};

TEST_P(SoftmaxAddReshapeTwoOutputsSubgraphTest, smoke_CompareWithRefs) {
    run();
}

namespace {

const std::vector<std::vector<InputShape>> inputShapes = {
    {
        // {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
        {{2, 64}, {{2, 64}}}
    },
    {
        // {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
        {{2, -1}, {{2, 64}}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Softmax_Add_Reshape_TwoOutputs, SoftmaxAddReshapeTwoOutputsSubgraphTest,
                        ::testing::ValuesIn(inputShapes),
                        SoftmaxAddReshapeTwoOutputsSubgraphTest::getTestCaseName);
} // namespace

class InputReshapeOutputSubgraphTest : virtual public ov::test::SubgraphBaseTest {
/*This test runs the following subgraph:

                      param
                        |
                        |
                    Reshape0
                        |
                        |
                     Result0

Edge Reshape0 -> Result0 cannot be referenced by its upstreams as its upstream is an input.
*/
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto precision = ov::element::f32;
        ov::test::InputShape input_shape{{-1, 16}, {{3, 16}}};
        init_input_shapes({input_shape});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, shape));
        }

        auto reshape_param_0 = ngraph::builder::makeConstant<int>(ov::element::i32, {1}, {0});
        auto reshape_0 = std::make_shared<ngraph::opset1::Unsqueeze>(params.front(), reshape_param_0);
        auto result_0 = std::make_shared<ngraph::opset3::Result>(reshape_0);    // target output

        ngraph::ResultVector results = {result_0};
        function = std::make_shared<ov::Model>(results, params, "Subgraph3");

        ov::pass::Serialize serializer("Subgraph3.xml", "Subgraph3.bin");
        serializer.run_on_model(function);
    }
};

TEST_F(InputReshapeOutputSubgraphTest, smoke_CompareWithRefs) {
    run();
}

} // namespace SubgraphTestsDefinitions