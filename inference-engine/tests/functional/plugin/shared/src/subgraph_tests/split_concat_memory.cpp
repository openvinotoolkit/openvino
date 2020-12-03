// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/split_concat_memory.hpp"
#include "common_test_utils/xml_net_builder/ir_net.hpp"

namespace LayerTestsDefinitions {

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

    auto input = std::make_shared<ngraph::op::Parameter>(ngPrc, ng_share_14);
    input->set_friendly_name("input");

    auto mem_c = std::make_shared<ngraph::op::Constant>(ngPrc, ng_share_34, 0);
    auto mem_r = std::make_shared<ngraph::op::ReadValue>(mem_c, "id");
    auto cnc = std::make_shared<ngraph::op::Concat>(ngraph::NodeVector{mem_r, input}, axis);

    std::vector<int64_t> chunks_val {static_cast<int64_t>(ng_share_14[axis]), static_cast<int64_t>(ng_share_34[axis])};
    auto chunk_c = std::make_shared<ngraph::op::Constant>(::ngraph::element::i64, ngraph::Shape{chunks_val.size()}, chunks_val);
    auto axis_c = std::make_shared<ngraph::op::Constant>(::ngraph::element::i64, ngraph::Shape{}, axis);
    auto spl = std::make_shared<ngraph::op::v1::VariadicSplit>(cnc, axis_c, chunk_c);

    auto one = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{}, 1);
    auto plus = std::make_shared<ngraph::op::v1::Add>(cnc, one, ngraph::op::AutoBroadcastSpec::NUMPY);
    plus->set_friendly_name("plus_one");

    auto mem_w = std::make_shared<ngraph::op::Assign>(spl->output(1), "id");

    // WA. Ngraph limitations. Assign should have control dependencies on read.
    // And someone should hold assign node.
    mem_w->add_control_dependency(mem_r);
    plus->add_control_dependency(mem_w);

    function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector      {plus},
            ngraph::ParameterVector {input},
            "CyclicBuffer4");
}

TEST_P(SplitConcatMemory, cyclicBufferCorrectness) {
    auto ie = PluginCache::get().ie();
    cnnNetwork = InferenceEngine::CNNNetwork{function};

    auto exe_net = ie->LoadNetwork(cnnNetwork, "CPU");
    auto inf_reg = exe_net.CreateInferRequest();

    /*
     * cnc1 out  |  mem      | In|q
     *           |===============|
     * iter_1    | 0 | 0 | 0 | 1 |
     * iter_2    | 0 | 0 | 1 | 2 |
     * iter 3    | 0 | 1 | 2 | 3 |
     */

    auto i_blob = inf_reg.GetBlob("input");
    auto o_blob = inf_reg.GetBlob("plus_one");

    auto o_blob_ref = make_blob_with_precision(o_blob->getTensorDesc());
    o_blob_ref->allocate();

    auto fill_by_quarter = [this] (Blob::Ptr& blob, std::vector<float> vals) {
        IE_ASSERT(vals.size() == 4);
        auto quarter_blocked_shape = blob->getTensorDesc().getDims();

        // splis axis dimension into chunk
        IE_ASSERT(quarter_blocked_shape[axis] % vals.size() == 0);
        quarter_blocked_shape[axis] /= vals.size();
        quarter_blocked_shape.insert(quarter_blocked_shape.begin() + axis, vals.size());

        auto quarter_blocked_view = make_reshape_view(blob, quarter_blocked_shape);
        fill_data_with_broadcast(quarter_blocked_view, axis, vals);
    };

    // iteration 1
    fill_data_const(i_blob, 1);
    fill_by_quarter(o_blob_ref, {1, 1, 1, 2});
    inf_reg.Infer();
    Compare(o_blob_ref, o_blob);

    // iteration 2
    fill_data_const(i_blob, 2);
    fill_by_quarter(o_blob_ref, {1, 1, 2, 3});
    inf_reg.Infer();
    Compare(o_blob_ref, o_blob);

    // iteration 3
    fill_data_const(i_blob, 3);
    fill_by_quarter(o_blob_ref, {1, 2, 3, 4});
    inf_reg.Infer();
    Compare(o_blob_ref, o_blob);
}

}  // namespace LayerTestsDefinitions