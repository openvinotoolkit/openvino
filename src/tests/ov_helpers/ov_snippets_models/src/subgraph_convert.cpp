// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_converts.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/op/convert_truncation.hpp>
#include <snippets/op/subgraph.hpp>

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Node> createRollAsStub(const std::shared_ptr<ov::Node>& parent) {
    auto shift = std::make_shared<op::v0::Constant>(ov::element::i32, Shape{1}, std::vector<int>{1});
    auto axes = std::make_shared<op::v0::Constant>(ov::element::i32, Shape{1}, std::vector<int>{0});
    return std::make_shared<op::v7::Roll>(parent->output(0), shift, axes);
}

std::shared_ptr<ov::Model> ConvertFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto convert = std::make_shared<op::v0::Convert>(data0, outType);
    return std::make_shared<ov::Model>(NodeVector{convert}, ParameterVector{data0});
}
std::shared_ptr<ov::Model> ConvertFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0}, getOriginal());
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data0});
}

std::shared_ptr<ov::Model> ConvertInputFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(outType, input_shapes[1]);
    auto convert = std::make_shared<op::v0::Convert>(data0, outType);
    auto add = std::make_shared<op::v1::Add>(convert, data1);
    return std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> ConvertInputFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(outType, input_shapes[1]);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0, data1}, getOriginal());
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> ConvertOutputFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inType, input_shapes[1]);
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    auto convert = std::make_shared<op::v0::Convert>(add, outType);
    return std::make_shared<ov::Model>(NodeVector{convert}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> ConvertOutputFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inType, input_shapes[1]);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0, data1}, getOriginal());
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> ConvertStubFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inType, input_shapes[1]);
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    auto convert = std::make_shared<op::v0::Convert>(add, outType);
    auto relu = std::make_shared<op::v0::Relu>(convert);
    return std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> ConvertStubFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inType, input_shapes[1]);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0, data1}, getOriginal());
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> ConvertPartialInputsAndResultsFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inTypes[0], input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inTypes[1], input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(inTypes[2], input_shapes[2]);
    auto convert0 = std::make_shared<op::v0::Convert>(data0, outTypes[0]);
    auto convert1 = std::make_shared<op::v0::Convert>(data1, outTypes[0]);
    auto add = std::make_shared<op::v1::Add>(convert0, convert1);
    auto relu = std::make_shared<op::v0::Relu>(add);
    auto sub = std::make_shared<op::v1::Subtract>(relu, data2);
    auto stub3 = createRollAsStub(sub);
    auto convert2 = std::make_shared<op::v0::Convert>(relu, outTypes[1]);
    return std::make_shared<ov::Model>(NodeVector{convert2, stub3}, ParameterVector{data0, data1, data2});
}
std::shared_ptr<ov::Model> ConvertPartialInputsAndResultsFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inTypes[0], input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inTypes[1], input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(inTypes[2], input_shapes[2]);
    auto indata0 = std::make_shared<op::v0::Parameter>(inTypes[0], data0->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(inTypes[1], data1->get_shape());
    auto indata2 = std::make_shared<op::v0::Parameter>(inTypes[2], data2->get_shape());
    auto convert0 = std::make_shared<op::v0::Convert>(indata0, outTypes[0]);
    auto convert1 = std::make_shared<op::v0::Convert>(indata1, outTypes[0]);
    auto add = std::make_shared<op::v1::Add>(convert0, convert1);
    auto relu = std::make_shared<op::v0::Relu>(add);
    auto sub = std::make_shared<op::v1::Subtract>(relu, indata2);
    auto convert2 = std::make_shared<op::v0::Convert>(relu, outTypes[1]);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(
            NodeVector{data0, data1, data2}, std::make_shared<ov::Model>(NodeVector{sub, convert2}, ParameterVector{indata0, indata1, indata2}));
    auto stub3 = createRollAsStub(subgraph);
    return std::make_shared<ov::Model>(OutputVector{subgraph->output(1), stub3->output(0)},
                                       ParameterVector{data0, data1, data2});
}

std::shared_ptr<ov::Model> ConvertManyOnInputsFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(types[0], input_shapes[0]);
    std::shared_ptr<ov::Node> out = data0;
    for (auto i = 1; i < types.size(); i++) {
        auto convert = std::make_shared<op::v0::Convert>(out, types[i]);
        out = convert;
    }
    auto relu = std::make_shared<op::v0::Relu>(out);
    return std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{data0});
}
std::shared_ptr<ov::Model> ConvertManyOnInputsFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(types[0], input_shapes[0]);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0}, getOriginal());
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data0});
}

std::shared_ptr<ov::Model> ConvertManyOnOutputsFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(types[0], input_shapes[0]);
    auto relu = std::make_shared<op::v0::Relu>(data0);
    std::shared_ptr<ov::Node> out = relu;
    for (auto i = 1; i < types.size(); i++) {
        auto convert = std::make_shared<op::v0::Convert>(out, types[i]);
        out = convert;
    }
    return std::make_shared<ov::Model>(NodeVector{out}, ParameterVector{data0});
}
std::shared_ptr<ov::Model> ConvertManyOnOutputsFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(types[0], input_shapes[0]);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0}, getOriginal());
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data0});
}

std::shared_ptr<ov::Model> ConvertManyOnInputOutputFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inTypes[0], input_shapes[0]);
    std::shared_ptr<ov::Node> out = data0;
    for (auto i = 1; i < inTypes.size(); i++) {
        auto convert = std::make_shared<op::v0::Convert>(out, inTypes[i]);
        out = convert;
    }
    auto relu = std::make_shared<op::v0::Relu>(data0);
    out = relu;
    for (auto i = 0; i < outTypes.size(); i++) {
        auto convert = std::make_shared<op::v0::Convert>(out, outTypes[i]);
        out = convert;
    }
    return std::make_shared<ov::Model>(NodeVector{out}, ParameterVector{data0});
}
std::shared_ptr<ov::Model> ConvertManyOnInputOutputFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inTypes[0], input_shapes[0]);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0}, getOriginal());
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data0});
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
