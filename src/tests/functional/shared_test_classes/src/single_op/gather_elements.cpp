// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/gather_elements.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/gather_elements.hpp"

namespace ov {
namespace test {
std::string GatherElementsLayerTest::getTestCaseName(const testing::TestParamInfo<GatherElementsParams>& obj) {
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type, indices_type;
    int axis;
    std::string device;
    std::tie(shapes, indices_shape, axis, model_type, indices_type, device) = obj.param;

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
    result << "IS=" << ov::test::utils::vec2str(indices_shape) << "_";
    result << "Ax=" << axis << "_";
    result << "DP=" << model_type.get_type_name() << "_";
    result << "IP=" << indices_type.get_type_name() << "_";
    result << "device=" << device;
    return result.str();
}

void GatherElementsLayerTest::SetUp() {
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type, indices_type;
    int axis;
    std::tie(shapes, indices_shape, axis, model_type, indices_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto axis_dim = targetStaticShapes[0][0][axis < 0 ? axis + targetStaticShapes[0][0].size() : axis];
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = axis_dim - 1;
    auto indices_node_tensor = ov::test::utils::create_and_fill_tensor(indices_type, indices_shape, in_data);
    auto indices_node = std::make_shared<ov::op::v0::Constant>(indices_node_tensor);

    auto gather_el = std::make_shared<ov::op::v6::GatherElements>(param, indices_node, axis);
    gather_el->set_friendly_name("GatherElements");

    auto result = std::make_shared<ov::op::v0::Result>(gather_el);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "gatherEl");
}
}  // namespace test
}  // namespace ov
