// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset5.hpp"
#include "shared_test_classes/subgraph/split_concat_memory.hpp"

namespace SubgraphTestsDefinitions {

using namespace CommonTestUtils;
using namespace InferenceEngine;

std::string SplitConcatMemory::getTestCaseName(testing::TestParamInfo<ParamType> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    int axis;
    std::string targetDevice;
    std::tie(inputShapes, netPrecision, axis, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "PRC=" << netPrecision.name() << "_";
    result << "axis=" << axis << "_";
    result << "dev=" << targetDevice;
    return result.str();
}

void SplitConcatMemory::SetUp() {
    SizeVector shape;
    std::tie(shape, inPrc, axis, targetDevice) = this->GetParam();

    auto shape_14 = shape;
    shape_14[axis] /= 4;
    auto shape_34 = shape;
    shape_34[axis] -= shape_14[axis];

    /*
     *    Cyclic buffer length of 4
     *        ______   ______
     *       [_mem1_] [_inp1_]
     *          _|______|_
     *         [_cocncat__]
     *         _____|______
     *      __|____     ___|__
     *     [_plus1_]   [_spl1_]
     *        |         |    |
     *      __|___         __|___
     *     [_out1_]       [_mem2_]
     */
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
    ngraph::Shape ng_share_14(shape_14);
    ngraph::Shape ng_share_34(shape_34);

    auto input = std::make_shared<ngraph::opset5::Parameter>(ngPrc, ng_share_14);
    input->set_friendly_name("input");

    auto mem_c = std::make_shared<ngraph::opset5::Constant>(ngPrc, ng_share_34, 0);
    auto mem_r = std::make_shared<ngraph::opset5::ReadValue>(mem_c, "id");
    auto cnc = std::make_shared<ngraph::opset5::Concat>(ngraph::NodeVector{mem_r, input}, axis);

    std::vector<int64_t> chunks_val {static_cast<int64_t>(ng_share_14[axis]), static_cast<int64_t>(ng_share_34[axis])};
    auto chunk_c = std::make_shared<ngraph::opset5::Constant>(::ngraph::element::i64, ngraph::Shape{chunks_val.size()}, chunks_val);
    auto axis_c = std::make_shared<ngraph::opset5::Constant>(::ngraph::element::i64, ngraph::Shape{}, axis);
    auto spl = std::make_shared<ngraph::opset5::VariadicSplit>(cnc, axis_c, chunk_c);

    auto one = std::make_shared<ngraph::opset5::Constant>(ngPrc, ngraph::Shape{}, 1);
    auto plus = std::make_shared<ngraph::opset5::Add>(cnc, one, ngraph::op::AutoBroadcastSpec::NUMPY);
    plus->set_friendly_name("plus_one");

    auto mem_w = std::make_shared<ngraph::opset5::Assign>(spl->output(1), "id");

    // WA. Ngraph limitations. Assign should have control dependencies on read.
    // And someone should hold assign node.
    mem_w->add_control_dependency(mem_r);
    plus->add_control_dependency(mem_w);

    function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector      {plus},
            ngraph::ParameterVector {input},
            "CyclicBuffer4");
}
}  // namespace SubgraphTestsDefinitions