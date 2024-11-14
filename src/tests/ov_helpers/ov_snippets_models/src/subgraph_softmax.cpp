// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_softmax.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include <snippets/op/subgraph.hpp>
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {
std::shared_ptr<ov::Node> buildSoftmax(const ov::Output<ov::Node>& input, int64_t axis, SoftmaxVersion softmax_version) {
    switch (softmax_version) {
        case SoftmaxVersion::V1: {
            OPENVINO_ASSERT(axis > 0, "v1::Softmax supports only positive axis.");
            return std::make_shared<ov::op::v1::Softmax>(input, static_cast<size_t>(axis));
        }
        case SoftmaxVersion::V8:
            return std::make_shared<ov::op::v8::Softmax>(input, axis);
        default:
            OPENVINO_THROW("Unexpected SoftmaxVersion.");
    }
}
}  // namespace

std::ostream &operator<<(std::ostream& os, const SoftmaxVersion& version) {
    switch (version) {
        case SoftmaxVersion::V1:
            return os << "V1";
        case SoftmaxVersion::V8:
            return os << "V8";
        default:
            OPENVINO_THROW("Unexpected SoftmaxVersion.");
    }
}

std::shared_ptr<ov::Model> SoftmaxFunction::initOriginal() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const auto softmax = buildSoftmax(data, axis, softmax_version);
    return std::make_shared<ov::Model>(NodeVector{softmax}, ParameterVector{data});
}

std::shared_ptr<ov::Model> SoftmaxFunction::initLowered() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const auto normalized_axis = ov::util::try_normalize_axis(axis, data->get_output_partial_shape(0).rank());

    const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(data, normalized_axis);
    const auto subtract = std::make_shared<ov::op::v1::Subtract>(data, reduce_max);
    const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);

    const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, normalized_axis);
    const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.f);
    const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);

    return std::make_shared<ov::Model>(NodeVector{multiply}, ParameterVector{data});
}

std::shared_ptr<ov::Model> AddSoftmaxFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto add = std::make_shared<ov::op::v1::Add>(data0, data1);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(add, axis);
    return std::make_shared<ov::Model>(NodeVector{softmax}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> TransposeSoftmaxFunction::initOriginal() const {
    const auto transpose0Param = std::make_shared<ov::opset1::Parameter>(precision, input_shapes[0]);
    const auto transpose0Const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{m_order.size()}, m_order);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto softMax = std::make_shared<ov::op::v8::Softmax>(transpose2, m_axis);
    return std::make_shared<ov::Model>(ov::NodeVector{softMax}, ov::ParameterVector {transpose0Param}, "softmax_transpose");
}

std::shared_ptr<ov::Model> TransposeSoftmaxEltwiseFunction::initOriginal() const {
    const auto transpose0Param = std::make_shared<ov::opset1::Parameter>(precision, input_shapes[0]);
    const auto transpose0Const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{m_order.size()}, m_order);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto mul1Param = std::make_shared<ov::opset1::Parameter>(precision, input_shapes[1]);
    const auto mul = std::make_shared<ov::op::v1::Multiply>(transpose2, mul1Param);
    const auto softMax = std::make_shared<ov::op::v8::Softmax>(mul, m_axis);
    const auto hswish = std::make_shared<ov::op::v4::HSwish>(softMax);
    return std::make_shared<ov::Model>(ov::NodeVector{hswish}, ov::ParameterVector{transpose0Param, mul1Param},
                                       "softmax_transpose");
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
