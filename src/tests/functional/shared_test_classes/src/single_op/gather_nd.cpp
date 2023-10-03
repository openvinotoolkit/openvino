// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/gather_nd.hpp"

#include "ngraph_functions/builders.hpp"

namespace ov {
namespace test {
std::string GatherNDLayerTest::getTestCaseName(const testing::TestParamInfo<GatherNDParams>& obj) {
    std::vector<InputShape> shapes;
    ov::Shape indices_shape;
    ov::element::Type model_type, indices_type;
    int batch_dims;
    std::string device;
    std::tie(shapes, indices_shape, batch_dims, model_type, indices_type, device) = obj.param;

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
    result << "BD=" << batch_dims << "_";
    result << "DP=" << model_type.get_type_name() << "_";
    result << "IP=" << indices_type.get_type_name() << "_";
    result << "device=" << device;
    return result.str();
}

void GatherNDLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::Shape indices_shape;
    ov::element::Type model_type, indices_type;
    int batch_dims;
    std::tie(shapes, indices_shape, batch_dims, model_type, indices_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto gather = ngraph::builder::makeGatherND(param, indices_shape, indices_type, batch_dims);

    auto result = std::make_shared<ov::op::v0::Result>(gather);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "gatherND");
}


std::string GatherND8LayerTest::getTestCaseName(const testing::TestParamInfo<GatherNDParams>& obj) {
    return GatherNDLayerTest::getTestCaseName(obj);
}

void GatherND8LayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::Shape indices_shape;
    ov::element::Type model_type, indices_type;
    int batch_dims;
    std::tie(shapes, indices_shape, batch_dims, model_type, indices_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto gather = ngraph::builder::makeGatherND8(param, indices_shape, indices_type, batch_dims);

    auto result = std::make_shared<ov::op::v0::Result>(gather);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "gatherND");
}
}  // namespace test
}  // namespace ov
