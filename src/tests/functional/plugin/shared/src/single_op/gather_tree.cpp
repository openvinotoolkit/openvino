// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/gather_tree.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/gather_tree.hpp"

namespace ov {
namespace test {
using ov::test::utils::InputLayerType;

std::string GatherTreeLayerTest::getTestCaseName(const testing::TestParamInfo<GatherTreeParamsTuple> &obj) {
    ov::Shape input_shape;
    ov::element::Type model_type;
    InputLayerType secondary_input_type;
    std::string device_name;

    std::tie(input_shape, secondary_input_type, model_type, device_name) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "secondary_input_type=" << secondary_input_type << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << device_name;
    return result.str();
}

void GatherTreeLayerTest::SetUp() {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    InputLayerType secondary_input_type;

    std::tie(input_shape, secondary_input_type, model_type, targetDevice) = GetParam();

    std::vector<ov::Shape> input_shapes_static {input_shape};
    std::vector<ov::Shape> constant_shapes_static;
    if (InputLayerType::PARAMETER == secondary_input_type) {
        input_shapes_static.push_back(input_shape);
        input_shapes_static.push_back(ov::Shape{input_shape.at(1)});
        input_shapes_static.push_back(ov::Shape());
    } else {
        constant_shapes_static.push_back(input_shape);
        constant_shapes_static.push_back(ov::Shape{input_shape.at(1)});
        constant_shapes_static.push_back(ov::Shape());
    }
    init_input_shapes(ov::test::static_shapes_to_test_representation(input_shapes_static));

    ov::ParameterVector params;
    ov::NodeVector inputs;
    for (const auto& shape : inputDynamicShapes) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, shape);
        params.push_back(param);
        inputs.push_back(param);
    }

    for (const auto& shape : constant_shapes_static) {
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 1;
        in_data.range = input_shape.at(2) - 2;
        auto tensor = ov::test::utils::create_and_fill_tensor(model_type, shape, in_data);
        auto constant = std::make_shared<ov::op::v0::Constant>(tensor);
        inputs.push_back(constant);
    }

    auto gt = std::make_shared<ov::op::v1::GatherTree>(inputs[0], inputs[1], inputs[2], inputs[3]);

    auto result = std::make_shared<ov::op::v0::Result>(gt);

    function = std::make_shared<ov::Model>(result, params, "GatherTree");
}
} // namespace test
} // namespace ov
