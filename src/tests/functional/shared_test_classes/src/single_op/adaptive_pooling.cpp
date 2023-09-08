// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/adaptive_pooling.hpp"

namespace ov {
namespace test {

std::string AdaPoolLayerTest::getTestCaseName(const testing::TestParamInfo<adapoolParams>& obj) {
    std::vector<InputShape> shapes;
    std::vector<int> pooled_spatial_shape;

    std::string pooling_mode;
    ov::element::Type type;
    std::string target_device;
    std::tie(shapes, pooled_spatial_shape, pooling_mode, type, target_device) = obj.param;

    std::ostringstream result;

    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "pooled_spatial_shape=" << ov::test::utils::vec2str(pooled_spatial_shape) << "_";
    result << "mode=" << pooling_mode << "_";
    result << "IT=" << type.get_type_name() << "_";
    result << "dev=" << target_device;
    return result.str();
}

void AdaPoolLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    std::vector<int> pooled_spatial_shape;
    std::string pooling_mode;
    std::tie(shapes, pooled_spatial_shape, pooling_mode, inType, targetDevice) = this->GetParam();
    init_input_shapes(shapes);
    outType = inType;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};

    ngraph::Shape pooled_shape = {pooled_spatial_shape.size()};
    auto pooled_param = std::make_shared<ov::op::v0::Constant>(ngraph::element::i32, pooled_shape, pooled_spatial_shape);

    // we cannot create abstract Op to use polymorphism
    auto adapoolMax = std::make_shared<ov::op::v8::AdaptiveMaxPool>(params[0], pooled_param, ngraph::element::i32);
    auto adapoolAvg = std::make_shared<ov::op::v8::AdaptiveAvgPool>(params[0], pooled_param);

    function = (pooling_mode == "max" ?
                std::make_shared<ov::Model>(adapoolMax->outputs(), params, "AdaPoolMax") :
                std::make_shared<ov::Model>(adapoolAvg->outputs(), params, "AdaPoolAvg"));
}
}  // namespace test
}  // namespace ov
