// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/broadcast.hpp"

#include "ngraph_functions/builders.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace LayerTestsDefinitions {
std::string BroadcastLayerTest::getTestCaseName(const testing::TestParamInfo<BroadcastParamsTuple>& obj) {
    InferenceEngine::SizeVector targetShape;
    ov::AxisSet axesMapping;
    ov::op::BroadcastType mode;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision networkPrecision;
    std::string deviceName;
    std::tie(targetShape, axesMapping, mode, inputShape, networkPrecision, deviceName) = obj.param;

    std::ostringstream result;
    result << "targetShape=" << ov::test::utils::vec2str(targetShape) << "_";
    result << "axesMapping=" << ov::test::utils::set2str(axesMapping)  << "_";
    result << "mode=" << mode << "_";
    result << "inShape=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "inNPrec=" << networkPrecision << "_";
    result << "trgDev=" << deviceName;
    return result.str();
}

void BroadcastLayerTest::SetUp() {
    InferenceEngine::SizeVector targetShape;
    ov::AxisSet axesMapping;
    ov::op::BroadcastType mode;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision networkPrecision;
    std::tie(targetShape, axesMapping, mode, inputShape, networkPrecision, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(networkPrecision);

    auto target_shape_const = ov::op::v0::Constant::create(ov::element::i64, {targetShape.size()}, targetShape);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto broadcast = ngraph::builder::makeBroadcast(params[0], target_shape_const, mode, axesMapping);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(broadcast)};
    function = std::make_shared<ov::Model>(results, params, "BroadcastInference");
}

}  // namespace LayerTestsDefinitions


namespace ov {
namespace test {

std::string BroadcastLayerTestNew::getTestCaseName(const testing::TestParamInfo<BroadcastParamsTuple>& obj) {
    ov::Shape targetShape;
    ov::AxisSet axesMapping;
    ov::op::BroadcastType mode;
    std::vector<InputShape> shapes;
    ov::element::Type type;
    std::string deviceName;
    std::tie(targetShape, axesMapping, mode, shapes, type, deviceName) = obj.param;

    std::ostringstream result;
    result << "targetShape=" << ov::test::utils::vec2str(targetShape) << "_";
    result << "axesMapping=" << ov::test::utils::set2str(axesMapping)  << "_";
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
    result << "trgDev=" << deviceName;
    return result.str();
}

void BroadcastLayerTestNew::SetUp() {
    std::vector<size_t> targetShape;
    ov::AxisSet axesMapping;
    ov::op::BroadcastType mode;
    std::vector<InputShape> shapes;
    std::tie(targetShape, axesMapping, mode, shapes, inType, targetDevice) = this->GetParam();
    init_input_shapes(shapes);
    outType = inType;
    if (inType == ElementType::bf16 || inType == ElementType::f16) {
        rel_threshold = 1e-2;
    }

    auto target_shape_const = ov::op::v0::Constant::create(ov::element::i64, {targetShape.size()}, targetShape);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};

    std::shared_ptr<ov::Node> broadcast;
    if (mode == ngraph::op::BroadcastType::NONE) {
        auto axisSetConst = ov::op::v0::Constant::create(ngraph::element::i64, {axesMapping.size()}, axesMapping.to_vector());
        broadcast = std::make_shared<ov::op::v3::Broadcast>(params[0],
                                                       target_shape_const,
                                                       axisSetConst,
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
