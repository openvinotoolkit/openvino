// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/simple_if.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"

namespace ov {
namespace test {

std::string SimpleIfTest::getTestCaseName(const testing::TestParamInfo<SimpleIfParamsTuple>& obj) {
    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType inType;
    bool condition;
    std::string targetName;
    std::tie(shapes, inType, condition, targetName) = obj.param;

    std::ostringstream results;
    for (size_t i = 0; i < shapes.size(); i++) {
        results << "Input" << i << "_";
        results << "IS=" << ov::test::utils::partialShape2str({shapes[i].first}) << "_";
        results << "TS=";
        for (const auto& item : shapes[i].second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
    }
    results << "inType=" << inType << "_";
    results << "Cond=" << condition << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void SimpleIfTest::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    // in bodies there aren't nodes that work with dimension 0. So we shouldn't call SubgraphBaseTest::compare
    bool hasZero = false;
    for (auto shape : targetStaticShapes[inferNum]) {
        hasZero = hasZero || std::any_of(shape.begin(), shape.end(), [](size_t dim) {
                      return dim == 0;
                  });
    }
    if (!hasZero) {
        SubgraphBaseTest::compare(expected, actual);
    }

    inferNum++;
}

void SimpleIfTest::SetUp() {
    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType inType;
    bool condition;
    std::tie(shapes, inType, condition, targetDevice) = this->GetParam();

    init_input_shapes(shapes);
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
    }

    auto p1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
    auto p2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]);
    auto p3 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);

    auto thenOp = std::make_shared<ov::op::v1::Add>(p1, p2);
    auto res1 = std::make_shared<ov::op::v0::Result>(thenOp);
    auto res2 = std::make_shared<ov::op::v0::Result>(p3);

    auto thenBody = std::make_shared<ov::Model>(ov::OutputVector{res1}, ov::ParameterVector{p1, p2});
    auto elseBody = std::make_shared<ov::Model>(ov::OutputVector{res2}, ov::ParameterVector{p3});

    auto condOp = ov::op::v0::Constant::create(ov::element::Type_t::boolean, {1}, std::vector<bool>{condition});
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
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
    }

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

    auto condOp = ov::op::v0::Constant::create(ov::element::Type_t::boolean, {1}, std::vector<bool>{condition});
    auto ifOp = std::make_shared<ov::op::v8::If>(condOp);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(params[0], p1, p3);
    ifOp->set_input(params[1], p2, p4);
    auto ifRes1 = ifOp->set_output(res1, res3);
    auto ifRes2 = ifOp->set_output(res2, res4);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ifRes1),
                             std::make_shared<ov::op::v0::Result>(ifRes2)};
    function = std::make_shared<ov::Model>(results, params, "simpleIf2Out");
}

void SimpleIfNotConstConditionTest::SetUp() {
    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType inType;
    std::tie(shapes, inType, condition, targetDevice) = this->GetParam();

    init_input_shapes(shapes);
    for (auto& target : targetStaticShapes)
        target.emplace_back(ov::Shape{});
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
    }
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

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ifRes1),
                             std::make_shared<ov::op::v0::Result>(ifRes2)};
    function = std::make_shared<ov::Model>(results, params, "SimpleIfNotConstConditionTest");
}

void SimpleIfNotConstConditionTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;

        if (i + 1 == funcInputs.size()) {
            tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            auto* dataPtr = tensor.data<bool>();
            dataPtr[0] = condition;
        } else {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -5;
            in_data.range = 10;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

void SimpleIfNotConstConditionAndInternalDynamismTest::SetUp() {
    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType inType;
    std::tie(shapes, inType, condition, targetDevice) = this->GetParam();

    init_input_shapes(shapes);
    for (auto& target : targetStaticShapes)
        target.emplace_back(ov::Shape{});
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
    }
    params.emplace_back(std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::boolean, ov::Shape{}));

    auto p1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
    auto p2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);

    // then body
    auto thenOp_0 = std::make_shared<ov::op::v3::NonZero>(p1, ov::element::i32);
    auto thenOp_1 = std::make_shared<ov::op::v0::Convert>(thenOp_0, inType);
    auto thenRes_0 = std::make_shared<ov::op::v0::Result>(p1);
    auto thenRes_1 = std::make_shared<ov::op::v0::Result>(thenOp_1);
    auto thenBody = std::make_shared<ov::Model>(ov::OutputVector{thenRes_0, thenRes_1}, ov::ParameterVector{p1});

    // else body
    auto add_const = std::make_shared<ov::op::v0::Constant>(inType, ov::Shape{}, std::vector<float>{2});
    auto elseOp_0 = std::make_shared<ov::op::v1::Add>(p2, add_const);
    auto elseOp_1 = std::make_shared<ov::op::v3::NonZero>(elseOp_0, ov::element::i32);
    auto elseOp_2 = std::make_shared<ov::op::v0::Convert>(elseOp_1, inType);
    auto elseRes_0 = std::make_shared<ov::op::v0::Result>(elseOp_0);
    auto elseRes_1 = std::make_shared<ov::op::v0::Result>(elseOp_2);
    auto elseBody = std::make_shared<ov::Model>(ov::OutputVector{elseRes_0, elseRes_1}, ov::ParameterVector{p2});

    auto ifOp = std::make_shared<ov::op::v8::If>(params[1]);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(params[0], p1, p2);
    auto ifRes_0 = ifOp->set_output(thenRes_0, elseRes_0);
    auto ifRes_1 = ifOp->set_output(thenRes_1, elseRes_1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ifRes_0),
                             std::make_shared<ov::op::v0::Result>(ifRes_1)};
    function = std::make_shared<ov::Model>(results, params, "SimpleIfNotConstConditionAndInternalDynamismTest");
}

void SimpleIfNotConstConditionAndDimsIncreaseTest::SetUp() {
    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType inType;
    std::tie(shapes, inType, condition, targetDevice) = this->GetParam();

    init_input_shapes(shapes);
    for (auto& target : targetStaticShapes)
        target.emplace_back(ov::Shape{});
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
    }
    params.emplace_back(std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::boolean, ov::Shape{}));

    auto p1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
    auto p2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);

    // then body
    const std::vector<int64_t> pads(p1->get_partial_shape().rank().get_length(), 2);
    auto pads_begin = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{pads.size()}, pads.data());
    auto pads_end = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{pads.size()}, pads.data());
    auto arg_pad_value = std::make_shared<ov::op::v0::Constant>(inType, ov::Shape{}, std::vector<int64_t>{0});
    auto thenOp = std::make_shared<ov::op::v1::Pad>(p1, pads_begin, pads_end, arg_pad_value, ov::op::PadMode::CONSTANT);

    auto thenRes = std::make_shared<ov::op::v0::Result>(thenOp);
    auto thenBody = std::make_shared<ov::Model>(ov::OutputVector{thenRes}, ov::ParameterVector{p1});

    // else body
    auto elseRes = std::make_shared<ov::op::v0::Result>(p2);
    auto elseBody = std::make_shared<ov::Model>(ov::OutputVector{elseRes}, ov::ParameterVector{p2});

    auto ifOp = std::make_shared<ov::op::v8::If>(params[1]);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(params[0], p1, p2);
    auto ifRes = ifOp->set_output(thenRes, elseRes);

    function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(ifOp)},
                                           params,
                                           "SimpleIfNotConstConditionAndDimsIncreaseTest");
}

void SimpleIfNotConstConditionAndDimsIncreaseTest::compare(const std::vector<ov::Tensor>& expected,
                                                           const std::vector<ov::Tensor>& actual) {
    const auto shape = targetStaticShapes[inferNum++].front();
    if (!condition && std::any_of(shape.begin(), shape.end(), [](size_t dim) {
            return dim == 0;
        })) {
        return;
    }

    SubgraphBaseTest::compare(expected, actual);
}

void SimpleIfNotConstConditionUnusedOutputPortsTest::SetUp() {
    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType inType;
    std::tie(shapes, inType, condition, targetDevice) = this->GetParam();

    init_input_shapes(shapes);
    for (auto& target : targetStaticShapes)
        target.emplace_back(ov::Shape{});
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
    }
    params.emplace_back(std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::boolean, ov::Shape{}));

    auto p1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
    auto p2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);

    const size_t axis = 1;
    const size_t dim = inputDynamicShapes[0][axis].get_length();  // should be static for this test suit
    auto thenOp_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{axis});
    auto thenOp = std::make_shared<ov::op::v1::Split>(p1, thenOp_axis_op, dim);
    auto thenRes = std::make_shared<ov::op::v0::Result>(thenOp->output(dim / 2));

    auto elseOp_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{axis});
    auto elseOp = std::make_shared<ov::op::v1::Split>(p2, elseOp_axis_op, dim);
    auto elseRes = std::make_shared<ov::op::v0::Result>(elseOp->output(dim - 1));

    auto thenBody = std::make_shared<ov::Model>(ov::OutputVector{thenRes}, ov::ParameterVector{p1});
    auto elseBody = std::make_shared<ov::Model>(ov::OutputVector{elseRes}, ov::ParameterVector{p2});

    auto ifOp = std::make_shared<ov::op::v8::If>(params[1]);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(params[0], p1, p2);
    auto ifRes = ifOp->set_output(thenRes, elseRes);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ifRes)};
    function = std::make_shared<ov::Model>(results, params, "SimpleIfNotConstConditionUnusedOutputPortsTest");
}

}  // namespace test
}  // namespace ov
