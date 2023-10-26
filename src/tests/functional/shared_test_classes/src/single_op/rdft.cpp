// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/rdft.hpp"
#include "ov_models/builders.hpp"

namespace ov {
namespace test {
std::string RDFTLayerTest::getTestCaseName(const testing::TestParamInfo<RDFTParams>& obj) {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::test::utils::DFTOpType op_type;
    std::string target_device;
    std::tie(input_shape, model_type, axes, signal_size, op_type, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "SignalSize=" << ov::test::utils::vec2str(signal_size) << "_";
    result << "Inverse=" << (op_type == ov::test::utils::DFTOpType::INVERSE) << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void RDFTLayerTest::SetUp() {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::test::utils::DFTOpType op_type;
    std::tie(input_shape, model_type, axes, signal_size, op_type, targetDevice) = this->GetParam();

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape));
    auto rdft = ngraph::builder::makeRDFT(param, axes, signal_size, op_type);
    function = std::make_shared<ov::Model>(rdft->outputs(), ov::ParameterVector{param}, "RDFT");
}
}  // namespace test
}  // namespace ov