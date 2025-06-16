// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/lrn.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/lrn.hpp"

namespace ov {
namespace test {
std::string LrnLayerTest::getTestCaseName(const testing::TestParamInfo<lrnLayerTestParamsSet>& obj) {
    double alpha, beta, bias;
    size_t size;
    std::vector<int64_t> axes;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::string targetDevice;
    std::tie(alpha, beta, bias, size, axes, model_type, shapes, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
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
    result << "Alpha=" << alpha << separator;
    result << "Beta=" << beta << separator;
    result << "Bias=" << bias << separator;
    result << "Size=" << size << separator;
    result << "Axes=" << ov::test::utils::vec2str(axes) << separator;
    result << "netPRC=" << model_type.get_type_name() << separator;
    result << "trgDev=" << targetDevice;

    return result.str();
}

void LrnLayerTest::SetUp() {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    double alpha, beta, bias;
    size_t size;
    std::vector<int64_t> axes;
    std::tie(alpha, beta, bias, size, axes, model_type, shapes, targetDevice) = GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto axes_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes.data());

    auto lrn = std::make_shared<ov::op::v0::LRN>(param, axes_node, alpha, beta, bias, size);

    auto result = std::make_shared<ov::op::v0::Result>(lrn);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "lrn");
}
}  // namespace test
}  // namespace ov
