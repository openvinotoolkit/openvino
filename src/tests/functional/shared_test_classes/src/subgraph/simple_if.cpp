// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/simple_if.hpp"
#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"

namespace SubgraphTestsDefinitions {
std::string SimpleIfTest::getTestCaseName(const testing::TestParamInfo<SimpleIfParamsTuple> &obj) {
    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType inType;
    bool condition;
    std::string targetName;
    std::tie(shapes, inType, condition, targetName) = obj.param;

    std::ostringstream results;
    for (size_t i = 0; i < shapes.size(); i++) {
        results << "Input" << i << "_";
        results << "IS=" << CommonTestUtils::partialShape2str({shapes[i].first}) << "_";
        results << "TS=";
        for (const auto &item : shapes[i].second) {
            results << CommonTestUtils::vec2str(item) << "_";
        }
    }
    results << "inType=" << inType << "_";
    results << "Cond=" << condition << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void SimpleIfTest::SetUp() {
    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType inType;
    bool condition;
    std::tie(shapes, inType, condition, targetDevice) = this->GetParam();

    init_input_shapes(shapes);
    auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);

    auto p1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
    auto p2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]);
    auto p3 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);

    auto thenOp = std::make_shared<ov::op::v1::Add>(p1, p2);
    auto res1 = std::make_shared<ov::op::v0::Result>(thenOp);
    auto res2 = std::make_shared<ov::op::v0::Result>(p3);

    auto thenBody = std::make_shared<ov::Model>(ov::OutputVector{res1}, ov::ParameterVector{p1, p2});
    auto elseBody = std::make_shared<ov::Model>(ov::OutputVector{res2}, ov::ParameterVector{p3});

    auto condOp = ngraph::builder::makeConstant<bool>(ov::element::Type_t::boolean, {1}, {condition});
    auto ifOp = std::make_shared<ov::op::v8::If>(condOp);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(params[0], p1, p3);
    ifOp->set_input(params[1], p2, nullptr);
    auto res = ifOp->set_output(res1, res2);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(res)};
    function = std::make_shared<ov::Model>(results, params, "simpleIf");
}

void SimpleIf2OutTest::SetUp() {
    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType inType;
    bool condition;
    std::tie(shapes, inType, condition, targetDevice) = this->GetParam();

    init_input_shapes(shapes);
    auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);

    auto p1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
    auto p2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]);
    auto p3 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
    auto p4 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]);

    auto thenOp = std::make_shared<ov::op::v1::Add>(p1, p2);
    auto res1 = std::make_shared<ov::op::v0::Result>(thenOp);
    auto res2 = std::make_shared<ov::op::v0::Result>(thenOp);
    auto res3 = std::make_shared<ov::op::v0::Result>(p3);
    auto res4 = std::make_shared<ov::op::v0::Result>(p4);

    auto thenBody = std::make_shared<ov::Model>(ov::OutputVector{res1, res2}, ov::ParameterVector{p1, p2});
    auto elseBody = std::make_shared<ov::Model>(ov::OutputVector{res3, res4}, ov::ParameterVector{p3, p4});

    auto condOp = ngraph::builder::makeConstant<bool>(ov::element::Type_t::boolean, {1}, {condition});
    auto ifOp = std::make_shared<ov::op::v8::If>(condOp);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(params[0], p1, p3);
    ifOp->set_input(params[1], p2, p4);
    auto ifRes1 = ifOp->set_output(res1, res3);
    auto ifRes2 = ifOp->set_output(res2, res4);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ifRes1), std::make_shared<ov::op::v0::Result>(ifRes2)};
    function = std::make_shared<ov::Model>(results, params, "simpleIf2Out");
}

void SimpleIfNotConstConditionTest::SetUp() {
    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType inType;
    std::tie(shapes, inType, condition, targetDevice) = this->GetParam();

    init_input_shapes(shapes);
    for (auto &target : targetStaticShapes)
        target.emplace_back(ov::Shape{});
    auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);
    params.emplace_back(std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::boolean, ov::Shape{}));

    auto p1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
    auto p2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]);
    auto p3 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
    auto p4 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]);

    auto thenOp = std::make_shared<ov::op::v1::Add>(p1, p2);
    auto res1 = std::make_shared<ov::op::v0::Result>(thenOp);
    auto res2 = std::make_shared<ov::op::v0::Result>(thenOp);
    auto res3 = std::make_shared<ov::op::v0::Result>(p3);
    auto res4 = std::make_shared<ov::op::v0::Result>(p4);

    auto thenBody = std::make_shared<ov::Model>(ov::OutputVector{res1, res2}, ov::ParameterVector{p1, p2});
    auto elseBody = std::make_shared<ov::Model>(ov::OutputVector{res3, res4}, ov::ParameterVector{p3, p4});

    auto ifOp = std::make_shared<ov::op::v8::If>(params[2]);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(params[0], p1, p3);
    ifOp->set_input(params[1], p2, p4);
    auto ifRes1 = ifOp->set_output(res1, res3);
    auto ifRes2 = ifOp->set_output(res2, res4);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ifRes1), std::make_shared<ov::op::v0::Result>(ifRes2)};
    function = std::make_shared<ov::Model>(results, params, "SimpleIfNotConstConditionTest");
}

void SimpleIfNotConstConditionTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::runtime::Tensor tensor;

        if (i + 1 == funcInputs.size()) {
            tensor = ov::runtime::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            auto *dataPtr = tensor.data<bool>();
            dataPtr[0] = condition;
        } else {
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 10, -5);
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

void SimpleIfNotConstConditionAndInternalDynamismTest::SetUp() {
    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType inType;
    std::tie(shapes, inType, condition, targetDevice) = this->GetParam();

    init_input_shapes(shapes);
    for (auto &target : targetStaticShapes)
        target.emplace_back(ov::Shape{});
    auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);
    params.emplace_back(std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::boolean, ov::Shape{}));

    auto p1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
    auto p2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);

    // then body
    auto thenOp_0 = std::make_shared<ov::op::v3::NonZero>(p1, ov::element::i32);
    auto thenOp_1 = std::make_shared<ov::op::v0::Convert>(thenOp_0, inType);
    auto thenRes = std::make_shared<ov::op::v0::Result>(thenOp_1);
    auto thenBody = std::make_shared<ov::Function>(ov::OutputVector{thenRes}, ov::ParameterVector{p1});

    // else body
    auto add_const = std::make_shared<ov::op::v0::Constant>(inType, ov::Shape{}, std::vector<float>{ 2 });
    auto elseOp_0 = std::make_shared<ov::op::v1::Add>(p2, add_const);
    auto elseOp_1 = std::make_shared<ov::op::v3::NonZero>(p2, ov::element::i32);
    auto elseOp_2 = std::make_shared<ov::op::v0::Convert>(elseOp_1, inType);
    auto elseRes = std::make_shared<ov::op::v0::Result>(elseOp_2);
    auto elseBody = std::make_shared<ov::Function>(ov::OutputVector{elseRes}, ov::ParameterVector{p2});

    auto ifOp = std::make_shared<ov::op::v8::If>(params[1]);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(params[0], p1, p2);
    auto ifRes = ifOp->set_output(thenRes, elseRes);

    function = std::make_shared<ov::Function>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(ifOp)},
                                              params, "SimpleIfNotConstConditionAndInternalDynamismTest");
}

} // namespace SubgraphTestsDefinitions
