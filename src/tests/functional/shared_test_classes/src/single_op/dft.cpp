// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/dft.hpp"

#include "common_test_utils/node_builders/dft.hpp"

namespace ov {
namespace test {
std::string DFTLayerTest::getTestCaseName(const testing::TestParamInfo<DFTParams>& obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::test::utils::DFTOpType op_type;
    std::string target_device;
    std::tie(shapes, model_type, axes, signal_size, op_type, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "Precision=" << model_type.get_type_name() << "_";
    result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "signal_size=" << ov::test::utils::vec2str(signal_size) << "_";
    result << "Inverse=" << (op_type == ov::test::utils::DFTOpType::INVERSE) << "_";
    result << "TargetDevice=" << target_device;
    return result.str();
}

void DFTLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::test::utils::DFTOpType op_type;
    std::tie(shapes, model_type, axes, signal_size, op_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto dft = ov::test::utils::make_dft(param, axes, signal_size, op_type);

    auto result = std::make_shared<ov::op::v0::Result>(dft);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "DFT");
}
}  // namespace test
}  // namespace ov
