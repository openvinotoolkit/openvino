// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/broadcast.hpp"

#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
std::string BroadcastLayerTest::getTestCaseName(const testing::TestParamInfo<BroadcastParamsTuple>& obj) {
    ov::Shape target_shape;
    ov::AxisSet axes_mapping;
    ov::op::BroadcastType mode;
    std::vector<InputShape> shapes;
    ov::element::Type type;
    std::string device_name;
    std::tie(target_shape, axes_mapping, mode, shapes, type, device_name) = obj.param;

    std::ostringstream result;
    result << "targetShape=" << ov::test::utils::vec2str(target_shape) << "_";
    result << "axesMapping=" << ov::test::utils::set2str(axes_mapping)  << "_";
    result << "mode=" << mode << "_";
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }    result << "IT=" << type.get_type_name() << "_";
    result << "trgDev=" << device_name;
    return result.str();
}

void BroadcastLayerTest::SetUp() {
    std::vector<size_t> target_shape;
    ov::AxisSet axes_mapping;
    ov::op::BroadcastType mode;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::tie(target_shape, axes_mapping, mode, shapes, model_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto target_shape_const = ov::op::v0::Constant::create(ov::element::i64, {target_shape.size()}, target_shape);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front())};

    std::shared_ptr<ov::Node> broadcast;
    if (mode == ov::op::BroadcastType::NONE) {
        auto axis_set_const = ov::op::v0::Constant::create(ov::element::i64, {axes_mapping.size()}, axes_mapping.to_vector());
        broadcast = std::make_shared<ov::op::v3::Broadcast>(params[0],
                                                            target_shape_const,
                                                            axis_set_const,
                                                            mode);
    } else { // numpy/bidirectional modes
        broadcast = std::make_shared<ov::op::v3::Broadcast>(params[0],
                                                            target_shape_const,
                                                            mode);
    }


    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(broadcast)};
    function = std::make_shared<ov::Model>(results, params, "BroadcastInference");
}
} //  namespace test
} //  namespace ov
