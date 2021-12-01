// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/loop.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"

using namespace InferenceEngine;
using namespace ov;
using namespace test;

namespace CPULayerTestsDefinitions {

enum LOOP_IN_TYPE {
    INVARIANT,
    MERGED
};

using LoopParams = typename std::tuple<
        bool,                                                              // ExecuteFirstIteration
        bool,                                                              // BodyCondition is a constant?
        bool,                                                              // BodyCondition value, if it is a Const
        int64_t,                                                           // TripCount, -1 means infinity
        std::vector<InputShape>,                                           // InputShapes
        std::vector<LOOP_IN_TYPE>,                                         // Type
        ElementType>;                                                      // Input element type


class LoopLayerCPUTest : public testing::WithParamInterface<LoopParams>,
                         virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LoopParams> obj) {
        bool execute_first_iteration;
        bool is_body_condition_const;
        bool body_condition; // works only if is_body_condition_const ==
        int64_t trip_count;
        std::vector<InputShape> shapes;
        std::vector<LOOP_IN_TYPE> types;
        ElementType netType;
        std::tie(execute_first_iteration, is_body_condition_const, body_condition, trip_count, shapes, types, netType) = obj.param;

        std::ostringstream result;
        for (size_t i = 0; i < shapes.size(); i++) {
            result << "Input" << i << "_";
            result << "IS=" << CommonTestUtils::partialShape2str({shapes[i].first}) << "_";
            result << "TS=";
            for (const auto& item : shapes[i].second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
            result << "types=" << types[i] << "_";
        }

        result << "execute_first_iteration" << execute_first_iteration << "_";
        result << "is_body_condition_const=" << is_body_condition_const << "_";
        result << "body_condition=" << body_condition << "_";
        result << "trip_count=" << trip_count << "_";
        result << "netType=" << netType;
        return result.str();
}

protected:
    void SetUp() override {
        bool execute_first_iteration;
        bool is_body_condition_const;
        bool body_condition; // works only if is_body_condition_const ==
        int64_t trip_count;
        std::vector<InputShape> shapes;
        std::vector<LOOP_IN_TYPE> types;
        ElementType netType;
        std::tie(execute_first_iteration, is_body_condition_const, body_condition, trip_count, shapes, types, netType) = this->GetParam();

        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes(shapes);

        auto params = ngraph::builder::makeDynamicParams(netType, inputDynamicShapes);

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        const std::vector<ngraph::PartialShape> body_params_shapes(shapes.size(), ngraph::PartialShape::dynamic());
        auto current_iteration = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{1});

        ngraph::ParameterVector body_params;
        for (const auto &pshape : body_params_shapes) {
            auto paramNode = std::make_shared<ngraph::opset1::Parameter>(netType, pshape);
            body_params.push_back(paramNode);
        }

        std::shared_ptr<ngraph::Node> body_condition_const;
        if (is_body_condition_const)
            body_condition_const = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{1}, body_condition);
        auto trip_count_const = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, trip_count);
        auto exec_condition = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{1}, execute_first_iteration);

        // Body
        std::shared_ptr<ngraph::Node> Zo = body_params[0];
        for (int i = 1; i < body_params.size(); ++i) {
            Zo = std::make_shared<ngraph::op::v1::Add>(body_params[i], Zo);
        }

        auto body = std::make_shared<ngraph::Function>(ngraph::OutputVector{body_condition_const, Zo},
                                                       body_params);

        auto loop = std::make_shared<ngraph::opset5::Loop>(trip_count_const, exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

        for (int i = 0; i < body_params.size(); ++i) {
            if (types[i] == LOOP_IN_TYPE::INVARIANT) {
                loop->set_invariant_input(body_params[i], params[i]);
            } else if (types[i] == LOOP_IN_TYPE::MERGED) {
                // todo: support several merged inputs
                // now supported only one in this sample
                loop->set_merged_input(body_params[i], params[i], Zo);
            }
        }

        // Output 0 is last Zo
        auto out0 = loop->get_iter_value(body_condition_const, -1);
        auto out1 = loop->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

        auto result0 = std::make_shared<ngraph::opset5::Result>(out0);
        auto result1 = std::make_shared<ngraph::opset5::Result>(out1);
        auto result2 = std::make_shared<ngraph::opset5::Result>(out2);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result0, result1, result2}, params, "loop");
    }
};

class LoopLayerForWithCondCPUTest : public testing::WithParamInterface<LoopParams>,
                                    virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LoopParams> obj) {
        return LoopLayerCPUTest::getTestCaseName(obj);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        // trip count
        {
            const auto& funcInput = funcInputs[0];
            ov::runtime::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                                 funcInput.get_shape(), 10, 1);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }

        // parameters for body
        for (int i = 1; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                                 targetInputStaticShapes[i], 15, 0, 32768);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        bool execute_first_iteration;
        bool is_body_condition_const;
        bool body_condition;
        int64_t trip_count; // doesn't work because it's dynamic
        std::vector<InputShape> shapes;
        std::vector<LOOP_IN_TYPE> types;
        ElementType netType;
        std::tie(execute_first_iteration, is_body_condition_const, body_condition, trip_count, shapes, types, netType) = this->GetParam();

        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes(shapes);
        for (auto& target : targetStaticShapes) {
            target.insert(target.begin(), ngraph::Shape{});
        }

        auto params = ngraph::builder::makeDynamicParams(netType, inputDynamicShapes);

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        const std::vector<ngraph::PartialShape> body_params_shapes(shapes.size(), ngraph::PartialShape::dynamic());
        auto current_iteration = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{1});

        ngraph::ParameterVector body_params;
        for (const auto &pshape : body_params_shapes) {
            auto paramNode = std::make_shared<ngraph::opset1::Parameter>(netType, pshape);
            body_params.push_back(paramNode);
        }

        std::shared_ptr<ngraph::Node> body_condition_const;
        if (is_body_condition_const)
            body_condition_const = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{1}, body_condition);
        auto trip_count_param = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{1});
        auto exec_condition = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{1}, execute_first_iteration);
        params.insert(params.begin(), trip_count_param);

        // Body
        std::shared_ptr<ngraph::Node> Zo = body_params[0];
        for (int i = 1; i < body_params.size(); ++i) {
            Zo = std::make_shared<ngraph::op::v1::Add>(body_params[i], Zo);
        }

        auto body = std::make_shared<ngraph::Function>(ngraph::OutputVector{body_condition_const, Zo}, body_params);
        auto loop = std::make_shared<ngraph::opset5::Loop>(params[0], exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

        for (int i = 0; i < body_params.size(); ++i) {
            if (types[i] == LOOP_IN_TYPE::INVARIANT) {
                loop->set_invariant_input(body_params[i], params[i + 1]);
            } else if (types[i] == LOOP_IN_TYPE::MERGED) {
                // todo: support several merged inputs
                // now supported only one in this sample
                loop->set_merged_input(body_params[i], params[i + 1], Zo);
            }
        }

        // Output 0 is last Zo
        auto out0 = loop->get_iter_value(body_condition_const, -1);
        auto out1 = loop->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

        auto result0 = std::make_shared<ngraph::opset5::Result>(out0);
        auto result1 = std::make_shared<ngraph::opset5::Result>(out1);
        auto result2 = std::make_shared<ngraph::opset5::Result>(out2);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result0, result1, result2}, params, "loop");
    }
};

TEST_P(LoopLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

TEST_P(LoopLayerForWithCondCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

const std::vector<ElementType> inputPrecisions = {
        ElementType::f32,
        ElementType::bf16,
        ElementType::i8
};

std::vector<bool> execute_first_iteration{true};
std::vector<bool> is_body_condition_const{true};
std::vector<bool> body_condition{true, false}; // works only if is_body_condition_const == true
std::vector<int64_t> trip_count{1, 5};
std::vector<InputShape> inputs = {
        {{-1, 1, -1}, {{10, 1, 10}, {1, 1, 1}, {1, 1, 1}, {5, 1, 3}}},
        {{-1, 1, -1}, {{1, 1, 1}, {5, 1, 2}, {5, 1, 2}, {5, 1, 3}}},
        {{-1, 1, -1}, {{10, 1, 10}, {5, 1, 2}, {5, 1, 2}, {5, 1, 3}}}
};
std::vector<LOOP_IN_TYPE> types = {
        LOOP_IN_TYPE::INVARIANT, LOOP_IN_TYPE::INVARIANT, LOOP_IN_TYPE::MERGED
};

INSTANTIATE_TEST_SUITE_P(smoke_LoopCommonZeroClip, LoopLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(execute_first_iteration),
                                 ::testing::ValuesIn(is_body_condition_const),
                                 ::testing::ValuesIn(body_condition),
                                 ::testing::ValuesIn(trip_count),
                                 ::testing::Values(inputs),
                                 ::testing::Values(types),
                                 ::testing::ValuesIn(inputPrecisions)),
                         LoopLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LoopCommonForWithCond, LoopLayerForWithCondCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(execute_first_iteration),
                                 ::testing::ValuesIn(is_body_condition_const),
                                 ::testing::ValuesIn(body_condition),
                                 ::testing::Values(trip_count.front()), // ignore
                                 ::testing::Values(inputs),
                                 ::testing::Values(types),
                                 ::testing::ValuesIn(inputPrecisions)),
                         LoopLayerForWithCondCPUTest::getTestCaseName);

}  // namespace
} // namespace CPULayerTestsDefinitions
