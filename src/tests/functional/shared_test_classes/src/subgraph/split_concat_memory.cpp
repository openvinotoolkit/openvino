// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/split_concat_memory.hpp"

namespace ov {
namespace test {

std::string SplitConcatMemory::getTestCaseName(const testing::TestParamInfo<ParamType>& obj) {
    ov::element::Type netPrecision;
    ov::Shape inputShapes;
    int axis;
    std::string targetDevice;
    std::tie(inputShapes, netPrecision, axis, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "PRC=" << netPrecision.get_type_name() << "_";
    result << "axis=" << axis << "_";
    result << "dev=" << targetDevice;
    return result.str();
}

void SplitConcatMemory::SetUp() {
    abs_threshold = 0.01;
    ov::Shape shape;

    std::tie(shape, inType, axis, targetDevice) = this->GetParam();

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
    ov::Shape ng_share_14(shape_14);
    ov::Shape ng_share_34(shape_34);

    auto input = std::make_shared<ov::op::v0::Parameter>(inType, ng_share_14);
    input->set_friendly_name("input");
    auto& tensor = input->get_output_tensor(0);
    tensor.set_names({"input_t"});
    // input->output(0).set_names({"input"});

    auto mem_c = std::make_shared<ov::op::v0::Constant>(inType, ng_share_34, 0);
    auto mem_r = std::make_shared<ov::op::v3::ReadValue>(mem_c, "id");
    auto cnc = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{mem_r, input}, axis);

    std::vector<int64_t> chunks_val{static_cast<int64_t>(ng_share_14[axis]), static_cast<int64_t>(ng_share_34[axis])};
    auto chunk_c = std::make_shared<ov::op::v0::Constant>(::ov::element::i64, ov::Shape{chunks_val.size()}, chunks_val);
    auto axis_c = std::make_shared<ov::op::v0::Constant>(::ov::element::i64, ov::Shape{}, axis);
    auto spl = std::make_shared<ov::op::v1::VariadicSplit>(cnc, axis_c, chunk_c);

    auto one = std::make_shared<ov::op::v0::Constant>(inType, ov::Shape{}, 1);
    auto plus = std::make_shared<ov::op::v1::Add>(cnc, one, ov::op::AutoBroadcastType::NUMPY);
    plus->set_friendly_name("plus_one");

    auto& o_tensor = plus->get_output_tensor(0);
    o_tensor.set_names({"plus_one_t"});
    // input->output(0).set_names({"plus_one"});

    auto mem_w = std::make_shared<ov::op::v3::Assign>(spl->output(1), "id");

    // WA. OpenVINO limitations. Assign should have control dependencies on read.
    // And someone should hold assign node.
    mem_w->add_control_dependency(mem_r);
    plus->add_control_dependency(mem_w);

    function = std::make_shared<ov::Model>(ov::NodeVector{plus}, ov::ParameterVector{input}, "CyclicBuffer4");
}

}  // namespace test
}  // namespace ov

