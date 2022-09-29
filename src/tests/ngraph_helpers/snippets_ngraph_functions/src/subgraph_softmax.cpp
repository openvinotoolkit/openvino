// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_softmax.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> SoftmaxFunction::initOriginal() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(data, axis);
    return std::make_shared<ov::Model>(NodeVector{softmax}, ParameterVector{data});
}

std::shared_ptr<ov::Model> SinhSoftmaxFunction::initOriginal() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto sinh = std::make_shared<ov::op::v0::Sinh>(data);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(sinh, axis);
    return std::make_shared<ov::Model>(NodeVector{softmax}, ParameterVector{data});
}

std::shared_ptr<ov::Model> AddSoftmaxFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto add = std::make_shared<ov::op::v1::Add>(data0, data1);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(add, axis);
    return std::make_shared<ov::Model>(NodeVector{softmax}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> SinhAddSoftmaxFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto sinh0 = std::make_shared<ov::op::v0::Sinh>(data0);
    auto sinh1 = std::make_shared<ov::op::v0::Sinh>(data1);
    auto add = std::make_shared<ov::op::v1::Add>(sinh0, sinh1);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(add, axis);
    return std::make_shared<ov::Model>(NodeVector{softmax}, ParameterVector{data0, data1});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov