// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/variadic_split_pad.hpp"


namespace ov {
namespace test {

std::string VariadicSplitPad::getTestCaseName(const testing::TestParamInfo<SplitPadTuple>& obj) {
    ov::Shape input_shape;
    int64_t axis;
    std::vector<size_t> numSplits, connectIndexes;
    std::vector<int64_t> padsBegin, padsEnd;
    ov::op::PadMode padMode;
    ov::element::Type element_type;
    std::string targetName;
    std::tie(input_shape, axis, numSplits, connectIndexes, padsBegin, padsEnd, padMode, element_type, targetName) =
        obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    results << "Axis=" << axis << "_";
    results << "NumSplits=" << ov::test::utils::vec2str(numSplits) << "_";
    results << "ConnectIndexes=" << ov::test::utils::vec2str(connectIndexes) << "_";
    results << "padsBegin=" << ov::test::utils::vec2str(padsBegin) << "_";
    results << "padsEnd=" << ov::test::utils::vec2str(padsEnd) << "_";
    results << "PadMode=" << padMode << "_";
    results << "netPRC=" << element_type << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void VariadicSplitPad::SetUp() {
    ov::Shape input_shape;
    int64_t axis;
    std::vector<size_t> numSplits, connectIndexes;
    std::vector<int64_t> padBegin, padEnd;
    ov::op::PadMode padMode;
    ov::element::Type element_type;
    std::tie(input_shape, axis, numSplits, connectIndexes, padBegin, padEnd, padMode, element_type, targetDevice) =
        this->GetParam();
    ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(input_shape))};

    auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, axis);
    auto num_split = std::make_shared<ov::op::v0::Constant>(ov::element::u64, ov::Shape{numSplits.size()}, numSplits);
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(input[0], split_axis_op, num_split);

    ov::ResultVector results;
    for (size_t i : connectIndexes) {
        auto pads_begin = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{padBegin.size()}, padBegin.data());
        auto pads_end = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{padEnd.size()}, padEnd.data());
        auto arg_pad_value = std::make_shared<ov::op::v0::Constant>(split->output(i).get_element_type(), ov::Shape{}, std::vector<int64_t>{0});
        auto pad = std::make_shared<ov::op::v1::Pad>(split->output(i), pads_begin, pads_end, arg_pad_value, padMode);

        results.push_back(std::make_shared<ov::op::v0::Result>(pad));
    }
    function = std::make_shared<ov::Model>(results, input, "variadic_split_pad");
}

}  // namespace test
}  // namespace ov
