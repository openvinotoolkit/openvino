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
    auto stub = createRollAsStub(data0);
    auto convert = std::make_shared<op::v0::Convert>(stub, outType);
    return std::make_shared<ov::Model>(NodeVector{convert}, ParameterVector{data0});
}
std::shared_ptr<ov::Model> ConvertFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto stub = createRollAsStub(data0);
    auto indata0 = std::make_shared<op::v0::Parameter>(inType, stub->get_shape());
    auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(NodeVector{stub},
         std::make_shared<ov::Model>(NodeVector{std::make_shared<ngraph::snippets::op::ConvertTruncation>(indata0, outType)},
                                     ParameterVector{indata0}));
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data0});
}

std::shared_ptr<ov::Model> ConvertInputFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(outType, input_shapes[1]);
    auto stub0 = createRollAsStub(data0);
    auto stub1 = createRollAsStub(data1);
    auto convert = std::make_shared<op::v0::Convert>(stub0, outType);
    auto add = std::make_shared<op::v1::Add>(convert, stub1);
    return std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> ConvertInputFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(outType, input_shapes[1]);
    auto stub0 = createRollAsStub(data0);
    auto stub1 = createRollAsStub(data1);
    auto indata0 = std::make_shared<op::v0::Parameter>(inType, stub0->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(outType, stub1->get_shape());
    auto convert = std::make_shared<ngraph::snippets::op::ConvertTruncation>(indata0, outType);
    auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(NodeVector{stub0, stub1},
        std::make_shared<ov::Model>(
                NodeVector{std::make_shared<op::v1::Add>(convert, indata1)},
                ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> ConvertOutputFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inType, input_shapes[1]);
    auto stub0 = createRollAsStub(data0);
    auto stub1 = createRollAsStub(data1);
    auto add = std::make_shared<op::v1::Add>(stub0, stub1);
    auto convert = std::make_shared<op::v0::Convert>(add, outType);
    return std::make_shared<ov::Model>(NodeVector{convert}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> ConvertOutputFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inType, input_shapes[1]);
    auto stub0 = createRollAsStub(data0);
    auto stub1 = createRollAsStub(data1);
    auto indata0 = std::make_shared<op::v0::Parameter>(inType, stub0->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(inType, stub1->get_shape());
    auto add = std::make_shared<op::v1::Add>(indata0, indata1);
    auto convert = std::make_shared<ngraph::snippets::op::ConvertTruncation>(add, outType);
    auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(NodeVector{stub0, stub1},
                                                                     std::make_shared<ov::Model>(
                                                                             NodeVector{convert},
                                                                             ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> ConvertStubFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inType, input_shapes[1]);
    auto stub0 = createRollAsStub(data0);
    auto stub1 = createRollAsStub(data1);
    auto add = std::make_shared<op::v1::Add>(stub0, stub1);
    auto convert = std::make_shared<op::v0::Convert>(add, outType);
    auto relu = std::make_shared<op::v0::Relu>(convert);
    return std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> ConvertStubFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inType, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inType, input_shapes[1]);
    auto stub0 = createRollAsStub(data0);
    auto stub1 = createRollAsStub(data1);
    auto indata0 = std::make_shared<op::v0::Parameter>(inType, stub0->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(inType, stub1->get_shape());
    auto add = std::make_shared<op::v1::Add>(indata0, indata1);
    auto convert = std::make_shared<ngraph::snippets::op::ConvertTruncation>(add, outType);
    auto subgraph0 = std::make_shared<ngraph::snippets::op::Subgraph>(
            NodeVector{stub0, stub1}, std::make_shared<ov::Model>(NodeVector{convert}, ParameterVector{indata0, indata1}));
    auto indata2 = std::make_shared<op::v0::Parameter>(convert->get_destination_type(), convert->get_shape());
    auto relu = std::make_shared<op::v0::Relu>(indata2);
    auto subgraph1 = std::make_shared<ngraph::snippets::op::Subgraph>(
            NodeVector{subgraph0}, std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{indata2}));
    return std::make_shared<ov::Model>(NodeVector{subgraph1}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> ConvertPartialInputsAndResultsFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inTypes[0], input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inTypes[1], input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(inTypes[2], input_shapes[2]);
    auto stub0 = createRollAsStub(data0);
    auto stub1 = createRollAsStub(data1);
    auto stub2 = createRollAsStub(data2);
    auto convert0 = std::make_shared<op::v0::Convert>(stub0, outTypes[0]);
    auto convert1 = std::make_shared<op::v0::Convert>(stub1, outTypes[0]);
    auto add = std::make_shared<op::v1::Add>(convert0, convert1);
    auto relu = std::make_shared<op::v0::Relu>(add);
    auto sub = std::make_shared<op::v1::Subtract>(relu, stub2);
    auto unsqueeze_axes = std::make_shared<op::v0::Constant>(ov::element::i32, Shape{}, std::vector<int>{1});
    auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(sub, unsqueeze_axes);
    auto convert2 = std::make_shared<op::v0::Convert>(relu, outTypes[1]);
    return std::make_shared<ov::Model>(NodeVector{convert2, unsqueeze}, ParameterVector{data0, data1, data2});
}
std::shared_ptr<ov::Model> ConvertPartialInputsAndResultsFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(inTypes[0], input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(inTypes[1], input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(inTypes[2], input_shapes[2]);
    auto stub0 = createRollAsStub(data0);
    auto stub1 = createRollAsStub(data1);
    auto stub2 = createRollAsStub(data2);
    auto indata0 = std::make_shared<op::v0::Parameter>(inTypes[0], stub0->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(inTypes[1], stub1->get_shape());
    auto indata2 = std::make_shared<op::v0::Parameter>(inTypes[2], stub2->get_shape());
    auto convert0 = std::make_shared<ngraph::snippets::op::ConvertTruncation>(indata0, outTypes[0]);
    auto convert1 = std::make_shared<ngraph::snippets::op::ConvertTruncation>(indata1, outTypes[0]);
    auto add = std::make_shared<op::v1::Add>(convert0, convert1);
    auto relu = std::make_shared<op::v0::Relu>(add);
    auto sub = std::make_shared<op::v1::Subtract>(relu, indata2);
    auto convert2 = std::make_shared<ngraph::snippets::op::ConvertTruncation>(relu, outTypes[1]);
    auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(
            NodeVector{stub0, stub1, stub2}, std::make_shared<ov::Model>(NodeVector{sub, convert2}, ParameterVector{indata0, indata1, indata2}));
    auto stub3 = createRollAsStub(subgraph);
    return std::make_shared<ov::Model>(OutputVector{subgraph->output(1), stub3->output(0)},
                                       ParameterVector{data0, data1, data2});
}
}  // namespace snippets
}  // namespace test
}  // namespace ov