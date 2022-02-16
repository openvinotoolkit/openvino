// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/op_impl_check/op_impl_check.hpp>
#include <single_layer_tests/op_impl_check/single_op_graph.hpp>

namespace ov {
namespace test {
namespace subgraph {

namespace {
std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::Op>& node) {
    return nullptr;
}

std::shared_ptr<ov::Model> generateBinaryEltwise(const std::shared_ptr<ov::op::Op>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 2}, {1, 2}});
    std::shared_ptr<ov::Node> eltwiseNode;
    if (ov::is_type<ov::op::v0::SquaredDifference>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::SquaredDifference>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Add>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Add>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Divide>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Divide>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::FloorMod>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::FloorMod>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Maximum>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Maximum>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Minimum>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Minimum>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Multiply>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Multiply>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Power>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Power>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Subtract>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Subtract>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Mod>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Mod>(params.front(), params.back());
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwiseNode)};
    return std::make_shared<ngraph::Function>(results, params, "BinaryEltwiseGraph");
}

std::shared_ptr<ov::Model> generateBinaryEltwiseComp(const std::shared_ptr<ov::op::Op>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2}, {2}});
    std::shared_ptr<ov::Node> comp;
    if (ov::is_type<ov::op::v1::Equal>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::EQUAL);
    } else if (ov::is_type<ov::op::v1::Greater>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::GREATER);
    } else if (ov::is_type<ov::op::v1::GreaterEqual>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::GREATER_EQUAL);
    } else if (ov::is_type<ov::op::v1::Less>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::LESS);
    } else if (ov::is_type<ov::op::v1::LessEqual>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::LESS_EQUAL);
    } else if (ov::is_type<ov::op::v1::NotEqual>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::NOT_EQUAL);
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(comp)};
    return std::make_shared<ngraph::Function>(results, params, "BinaryEltwiseComparisonGraph");
}

std::shared_ptr<ov::Model> generateBinaryEltwiseLogical(const std::shared_ptr<ov::op::Op>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::boolean, {{1, 2}, {1, 2}});
    std::shared_ptr<ov::Node> eltwiseNode;
    if (ov::is_type<ov::op::v1::LogicalAnd>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::LogicalAnd>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v1::LogicalOr>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::LogicalOr>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v1::LogicalXor>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::LogicalXor>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v0::Xor>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Xor>(params[0], params[1]);
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwiseNode)};
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{params}, "BinaryEltwiseLogicalGraph");
}

std::shared_ptr<ov::Model> generateBroadcast(const std::shared_ptr<ov::op::Op>& node) {
    const ov::Shape input_shape{};
    const ov::Shape output_shape{5, 4, 3, 2};
    const auto params = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    const auto shape_const =
        ov::op::v0::Constant::create(ov::element::u64, ov::Shape{output_shape.size()}, output_shape);
    std::shared_ptr<ov::Node> broadcast;
    if (ov::is_type<ov::op::v1::Broadcast>(node)) {
        broadcast = std::make_shared<ov::op::v1::Broadcast>(params, shape_const);
    } else if (ov::is_type<ov::op::v3::Broadcast>(node)) {
        broadcast = std::make_shared<ov::op::v3::Broadcast>(params, shape_const);
    } else {
        return nullptr;
    }

    return std::make_shared<ngraph::Function>(broadcast, ParameterVector{params}, "BroadcastGraph");
}

std::shared_ptr<ov::Model> generateConvertColor(const std::shared_ptr<ov::op::Op>& node) {
    const auto params = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, Shape{1, 3, 2, 1});
    std::shared_ptr<ov::Node> convert;
    if (ov::is_type<ov::op::v8::NV12toBGR>(node)) {
        convert = std::make_shared<ov::op::v8::NV12toBGR>(params);
    } else if (ov::is_type<ov::op::v8::NV12toRGB>(node)) {
        convert = std::make_shared<ov::op::v8::NV12toRGB>(params);
    } else if (ov::is_type<ov::op::v8::I420toBGR>(node)) {
        convert = std::make_shared<ov::op::v8::I420toBGR>(params);
    } else if (ov::is_type<ov::op::v8::I420toRGB>(node)) {
        convert = std::make_shared<ov::op::v8::I420toRGB>(params);
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(convert)};
    return std::make_shared<ngraph::Function>(results, ParameterVector{params}, "ConvertColorGraph");
}

std::shared_ptr<ov::Model> generateMultiSubGraph(const std::shared_ptr<ov::op::Op>& node) {
    if (ov::is_type<ov::op::v8::If>(node)) {
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, Shape{1});
        auto A = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 8.0);
        auto B = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, 2.0);
        auto A_res = std::make_shared<ov::op::v0::Result>(A);
        auto B_res = std::make_shared<ov::op::v0::Result>(B);
        auto then_body = std::make_shared<ov::Model>(OutputVector{A_res}, ParameterVector{});
        auto else_body = std::make_shared<ov::Model>(OutputVector{B_res}, ParameterVector{});
        auto if_op = std::make_shared<ov::op::v8::If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        auto res = if_op->set_output(A_res, B_res);
        return std::make_shared<ngraph::Function>(OutputVector{res}, ParameterVector{cond}, "MultiSubGraphOp");
    } else {
        return nullptr;
    }
}

std::shared_ptr<ov::Model> generateNmsBase(const std::shared_ptr<ov::op::Op>& node) {
    const auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    std::shared_ptr<ov::Node> nms;
    if (ov::is_type<ov::op::v8::MulticlassNms>(node)) {
        nms = std::make_shared<ov::op::v8::MulticlassNms>(boxes, scores, ov::op::v8::MulticlassNms::Attributes());
    } else if (ov::is_type<ov::op::v8::MatrixNms>(node)) {
        nms = std::make_shared<ov::op::v8::MatrixNms>(boxes, scores, ov::op::v8::MatrixNms::Attributes());
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(nms->output(0)),
                                 std::make_shared<ngraph::opset1::Result>(nms->output(1)),
                                 std::make_shared<ngraph::opset1::Result>(nms->output(2))};
    return std::make_shared<ngraph::Function>(results, ParameterVector{boxes, scores}, "NmsBase");
}

std::shared_ptr<ov::Model> generateReadValueAssignBase(const std::shared_ptr<ov::op::Op>& node) {
    auto in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    if (ov::is_type<ov::op::v3::ReadValue>(node) || ov::is_type<ov::op::v3::Assign>(node)) {
        auto read_value = std::make_shared<ov::op::v3::ReadValue>(in, "v0");
        auto assign = std::make_shared<ov::op::v3::Assign>(read_value, "v0");
        return std::make_shared<ov::Model>(OutputVector{assign}, ParameterVector{in}, "ReadValue_Assign");
    } else if (ov::is_type<ov::op::v6::ReadValue>(node) || ov::is_type<ov::op::v6::Assign>(node)) {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "v0"});
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(in, variable);
        auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);
        return std::make_shared<ov::Model>(OutputVector{assign},
                                           ParameterVector{in},
                                           ov::op::util::VariableVector{variable}, "ReadValue_Assign");
    } else {
        return nullptr;
    }
}
}  // namespace

template <typename T>
std::shared_ptr<ov::Model> generateGraph() {
    std::shared_ptr<T> node = std::shared_ptr<T>(new T);
    if (ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(node)) {
        return generateBinaryEltwise(node);
    } else if (ov::is_type<ov::op::util::BinaryElementwiseComparison>(node)) {
        return generateBinaryEltwiseComp(node);
    } else if (ov::is_type<ov::op::util::BinaryElementwiseLogical>(node)) {
        return generateBinaryEltwiseLogical(node);
    } else if (ov::is_type<ov::op::util::BroadcastBase>(node)) {
        return generateBroadcast(node);
    } else if (ov::is_type<ov::op::util::ConvertColorNV12Base>(node) ||
               ov::is_type<ov::op::util::ConvertColorI420Base>(node)) {
        return generateConvertColor(node);
    } else if (ov::is_type<ov::op::util::MultiSubGraphOp>(node)) {
        return generateMultiSubGraph(node);
    } else if (ov::is_type<ov::op::util::NmsBase>(node)) {
        return generateNmsBase(node);
    } else if (ov::is_type<ov::op::util::AssignBase>(node) || ov::is_type<ov::op::util::ReadValueBase>(node)) {
        return generateReadValueAssignBase(node);
    }

    return generate(node);
}

OpGenerator getOpGeneratorMap() {
    static OpGenerator opGeneratorMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), generateGraph<NAMESPACE::NAME>},
#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#undef _OPENVINO_OP_REG
    };
    return opGeneratorMap;
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov